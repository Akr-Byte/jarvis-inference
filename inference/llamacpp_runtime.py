"""Lifecycle and capability management for an owned llama-server child."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import httpx
import psutil

from inference.config import settings
from inference.gguf import estimate_kv_memory, read_metadata
from inference.model_registry import discover_gguf
from inference.resources import memory_available_gb

logger = logging.getLogger(__name__)

OPTIONAL_FLAGS = {
    "--alias": "alias",
    "--cache-prompt": "prompt_cache",
    "--cache-ram": "cache_ram_limit",
    "--cache-reuse": "cache_reuse",
    "--cache-type-k": "kv_cache_type_k",
    "--cache-type-v": "kv_cache_type_v",
    "--draft-model": "draft_model",
    "--draft-max-len": "draft_max_len",
    "--draft-min-len": "draft_min_len",
    "--draft-ctx-size": "draft_ctx_size",
    "--flash-attn": "flash_attention",
    "--jinja": "jinja",
    "--metrics": "metrics",
    "--mlock": "mlock",
    "--n-gpu-layers": "gpu_layers",
    "--parallel": "parallel_slots",
    "--reasoning-format": "reasoning_format",
    "--threads-batch": "batch_threads",
    "--embedding": "embeddings",
}
# Cache types that llama.cpp treats as an f16 cache (no quantization).
UNQUANTIZED_CACHE_TYPES = frozenset({"", "f16", "f32", "bf16"})


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class MemoryEstimate:
    components_bytes: dict[str, int]
    estimated_bytes: int
    headroom_bytes: int
    capacity_bytes: int
    complete: bool
    notes: tuple[str, ...]


def estimate_runtime_memory(
    *,
    model_bytes: int,
    metadata: dict[str, object],
    context_tokens: int,
    parallel_sequences: int,
    batch_size: int,
    ubatch_size: int,
    cache_ram_bytes: int,
    prompt_cache_enabled: bool,
    draft_model_bytes: int = 0,
) -> MemoryEstimate:
    if min(model_bytes, context_tokens, parallel_sequences, batch_size, ubatch_size) <= 0:
        raise ValueError("model size, context, parallelism, and batch sizes must be positive")
    if ubatch_size > batch_size:
        raise ValueError("micro-batch size cannot exceed batch size")
    if cache_ram_bytes < 0:
        raise ValueError("prompt cache RAM limit cannot be negative")
    architecture = str(metadata.get("general.architecture") or "unknown")
    hidden = metadata.get(f"{architecture}.embedding_length")
    if not isinstance(hidden, int) or hidden <= 0:
        raise ValueError("GGUF embedding_length metadata is required for memory admission")
    kv = estimate_kv_memory(
        metadata,
        context_tokens,
        parallel_sequences,
        settings.llamacpp_cache_type_k or "f16",
        settings.llamacpp_cache_type_v or settings.llamacpp_cache_type_k or "f16",
    )
    if not kv.complete or kv.bytes_total is None:
        raise ValueError(f"cannot safely estimate {architecture} KV/recurrent memory: {kv.reason}")
    load_slack_bytes = int(model_bytes * 0.10)
    batch_bytes = int(batch_size * hidden * 4 * 2)
    compute_bytes = int(ubatch_size * hidden * 4 * 4 + context_tokens * hidden * 4)
    cache_bytes = (
        cache_ram_bytes if cache_ram_bytes else (kv.bytes_total if prompt_cache_enabled else 0)
    )
    runtime_bytes = max(512 * 1024**2, int((model_bytes + load_slack_bytes) * 0.05))
    components = {
        "model_weights": model_bytes,
        "model_load_alignment_reserve": load_slack_bytes,
        "kv_and_recurrent_state": kv.bytes_total,
        "prompt_cache_reserve": cache_bytes,
        "batch_buffers": batch_bytes,
        "compute_buffers": compute_bytes,
        "runtime_overhead": runtime_bytes,
    }
    draft_bytes = int(draft_model_bytes) if draft_model_bytes and draft_model_bytes > 0 else 0
    if draft_bytes:
        components["draft_weights"] = draft_bytes
    components["uncertainty_reserve"] = max(1024**3, int(sum(components.values()) * 0.20))
    incomplete_notes = (
        "compute buffers and runtime overhead are conservative estimates",
        "prompt-cache reserve covers one context unless LLAMACPP_CACHE_RAM_MIB sets a larger bound",
        "GGUF file size does not establish actual resident process memory",
    )
    estimate = sum(components.values())
    return MemoryEstimate(
        components,
        estimate,
        int(settings.memory_headroom_gb * 1024**3),
        int(min(memory_available_gb(), settings.memory_budget_gb) * 1024**3),
        False,
        incomplete_notes,
    )


class LlamaCppRuntime:
    """Own only child processes created here; never signal port occupants or adopt PIDs."""

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self._log = None
        self._log_path: Path | None = None
        self._log_size = 0
        self._log_task: asyncio.Task[None] | None = None
        self._monitor: asyncio.Task[None] | None = None
        self._stopping = False
        self._lock = asyncio.Lock()
        self.state = RuntimeState.STOPPED
        self.last_error: str | None = None
        self.last_return_code: int | None = None
        self.restart_count = 0
        self._recovery_attempts = 0
        self.supported_flags: set[str] | None = None
        self._capability_probe_error: str | None = None
        self.model_memory_estimate_gb: float | None = None
        self.memory_estimate: MemoryEstimate | None = None
        self.model_context_capability: int | None = None
        self.active_model_id: str | None = None
        self.active_model_path: str | None = None
        self.executable_path: str | None = None

    async def detect_capabilities(self) -> dict[str, object]:
        configured_executable = Path(settings.llamacpp_server).expanduser()
        executable = (
            shutil.which(settings.llamacpp_server)
            if "/" not in settings.llamacpp_server
            else str(configured_executable.resolve())
        )
        if not executable or not Path(executable).is_file():
            raise RuntimeError(
                f"llama-server executable not found: {settings.llamacpp_server!r}; "
                "install llama.cpp or set LLAMACPP_SERVER to its executable path"
            )
        try:
            probe = await asyncio.create_subprocess_exec(
                executable,
                "--help",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await asyncio.wait_for(probe.communicate(), timeout=10)
        except TimeoutError as exc:
            if "probe" in locals() and probe.returncode is None:
                probe.kill()
                await probe.wait()
            raise RuntimeError(
                "llama-server --help timed out while detecting runtime options"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"unable to execute llama-server at {executable!r}: {exc}") from exc
        help_text = output.decode("utf-8", errors="replace")
        if probe.returncode not in (0, 1) or "--model" not in help_text:
            raise RuntimeError(
                f"could not read llama-server options (exit {probe.returncode}); "
                "check LLAMACPP_SERVER and its permissions"
            )
        self.supported_flags = {
            flag
            for flag in (
                "--model",
                "--host",
                "--port",
                "--ctx-size",
                "--threads",
                "--batch-size",
                "--ubatch-size",
                *OPTIONAL_FLAGS,
            )
            if flag in help_text
        }
        required = {
            "--model",
            "--host",
            "--port",
            "--ctx-size",
            "--threads",
            "--batch-size",
            "--ubatch-size",
        }
        missing = sorted(required - self.supported_flags)
        if missing:
            raise RuntimeError(
                "installed llama-server is missing required options: " + ", ".join(missing)
            )
        self.executable_path = executable
        return {
            "executable": executable,
            "supported_optional": {
                capability: flag in self.supported_flags
                for flag, capability in OPTIONAL_FLAGS.items()
            },
            "probe_error": self._capability_probe_error,
        }

    def command(self) -> list[str]:
        model_path = self.active_model_path or settings.llamacpp_model
        if not model_path:
            raise RuntimeError(
                "LLAMACPP_MODEL must point to a GGUF file when LLAMACPP_MANAGED=true"
            )
        cmd = [
            self.executable_path or settings.llamacpp_server,
            "--model",
            model_path,
            "--host",
            settings.llamacpp_host,
            "--port",
            str(settings.llamacpp_port),
            "--ctx-size",
            str(settings.max_context_length * settings.llamacpp_parallel),
            "--threads",
            str(settings.llamacpp_threads),
            "--batch-size",
            str(settings.llamacpp_batch_size),
            "--ubatch-size",
            str(settings.llamacpp_ubatch_size),
        ]
        switches: list[tuple[str, bool, str]] = [
            ("--cache-prompt", settings.llamacpp_cache_prompt, "prompt_cache"),
            ("--jinja", settings.llamacpp_jinja, "jinja"),
            ("--metrics", settings.llamacpp_metrics, "metrics"),
            ("--mlock", settings.llamacpp_mlock, "mlock"),
            ("--embedding", settings.llamacpp_embeddings, "embeddings"),
        ]
        reasoning_format = settings.llamacpp_reasoning_format
        if reasoning_format and reasoning_format not in {"auto", "none", "deepseek"}:
            logger.warning(
                "ignoring unsupported LLAMACPP_REASONING_FORMAT=%r; use auto, none or deepseek",
                reasoning_format,
            )
            reasoning_format = ""
        valued: list[tuple[str, str | None, str]] = [
            ("--alias", self.active_model_id or settings.llamacpp_alias or None, "alias"),
            (
                "--cache-ram",
                str(settings.llamacpp_cache_ram_mib * 1024**2)
                if settings.llamacpp_cache_ram_mib
                else None,
                "cache_ram_limit",
            ),
            (
                "--cache-reuse",
                str(settings.llamacpp_cache_reuse) if settings.llamacpp_cache_reuse else None,
                "cache_reuse",
            ),
            ("--cache-type-k", settings.llamacpp_cache_type_k or None, "kv_cache_type_k"),
            ("--cache-type-v", settings.llamacpp_cache_type_v or None, "kv_cache_type_v"),
            (
                "--flash-attn",
                "auto" if settings.llamacpp_flash_attention else None,
                "flash_attention",
            ),
            (
                "--n-gpu-layers",
                str(settings.llamacpp_gpu_layers) if settings.llamacpp_gpu_layers else None,
                "gpu_layers",
            ),
            ("--parallel", str(settings.llamacpp_parallel), "parallel_slots"),
            ("--reasoning-format", reasoning_format or None, "reasoning_format"),
            ("--threads-batch", str(settings.llamacpp_batch_threads), "batch_threads"),
        ]
        for flag, enabled, capability in switches:
            if not enabled or not self._flag_supported(flag, capability):
                continue
            cmd.append(flag)
        for flag, value, capability in valued:
            if value is None or not self._flag_supported(flag, capability):
                continue
            cmd.extend((flag, value))
        if settings.llamacpp_draft_model:
            if self._flag_supported("--draft-model", "draft_model"):
                cmd.extend(("--draft-model", settings.llamacpp_draft_model))
            if self._flag_supported("--draft-max-len", "draft_max_len"):
                cmd.extend(("--draft-max-len", str(settings.llamacpp_draft_max_len)))
            if self._flag_supported("--draft-min-len", "draft_min_len"):
                cmd.extend(
                    (
                        "--draft-min-len",
                        str(min(settings.llamacpp_draft_min_len, settings.llamacpp_draft_max_len)),
                    )
                )
            if settings.llamacpp_draft_ctx_size and self._flag_supported(
                "--draft-ctx-size", "draft_ctx_size"
            ):
                cmd.extend(("--draft-ctx-size", str(settings.llamacpp_draft_ctx_size)))
        return cmd

    def _flag_supported(self, flag: str, capability: str) -> bool:
        if self.supported_flags is None or flag in self.supported_flags:
            return True
        logger.warning(
            "llama-server does not support configured optional feature %s (%s); skipping",
            capability,
            flag,
        )
        return False

    async def start(self, model_id: str | None = None) -> None:
        if not settings.llamacpp_managed:
            return
        async with self._lock:
            self._stopping = False
            if self.process and self.process.returncode is None:
                if model_id is None or model_id == self.active_model_id:
                    return
                raise RuntimeError(
                    "stop the currently loaded model before switching runtime profiles"
                )
            expected_url = f"http://{settings.llamacpp_host}:{settings.llamacpp_port}"
            if settings.llamacpp_url and settings.llamacpp_url != expected_url:
                raise RuntimeError(
                    "LLAMACPP_URL must match LLAMACPP_HOST and LLAMACPP_PORT in managed mode"
                )
            self.state = RuntimeState.STARTING
            self.last_error = None
            self._recovery_attempts = 0
            self.model_memory_estimate_gb = None
            self.memory_estimate = None
            self.model_context_capability = None
            try:
                if model_id is not None:
                    self.active_model_id = model_id
                    self.active_model_path = settings.llamacpp_model_map.get(model_id)
                    if self.active_model_path is None and model_id == (
                        settings.llamacpp_alias or Path(settings.llamacpp_model).expanduser().stem
                    ):
                        self.active_model_path = settings.llamacpp_model
                    if self.active_model_path is None:
                        raise RuntimeError(
                            f"no managed llama.cpp model profile is configured for {model_id!r}"
                        )
                elif self.active_model_path is None:
                    selected = (
                        settings.default_model
                        if settings.default_model in settings.llamacpp_model_map
                        else None
                    )
                    selected = selected or next(iter(settings.llamacpp_model_map), None)
                    self.active_model_id = selected or settings.llamacpp_alias or None
                    self.active_model_path = (
                        settings.llamacpp_model_map.get(selected, settings.llamacpp_model)
                        if selected
                        else settings.llamacpp_model
                    )
                await self.detect_capabilities()
                if settings.llamacpp_embeddings and "--embedding" not in (
                    self.supported_flags or set()
                ):
                    raise RuntimeError(
                        "LLAMACPP_EMBEDDINGS=true but this llama-server build does not support --embedding"
                    )
                model_path = Path(self.active_model_path or "").expanduser()
                if not model_path.is_file():
                    raise RuntimeError(f"configured GGUF model does not exist: {model_path}")
                self.active_model_path = str(model_path.resolve())
                try:
                    metadata = read_metadata(model_path)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"unable to read GGUF metadata from {model_path}: {exc}"
                    ) from exc
                architecture = str(metadata.get("general.architecture") or "llama")
                trained_context = metadata.get(f"{architecture}.context_length")
                self.model_context_capability = (
                    int(trained_context)
                    if isinstance(trained_context, int) and trained_context > 0
                    else None
                )
                if (
                    self.model_context_capability
                    and settings.max_context_length > self.model_context_capability
                ):
                    raise RuntimeError(
                        f"configured context {settings.max_context_length} exceeds the model's "
                        f"trained context {self.model_context_capability}"
                    )
                if settings.max_context_length < 1 or settings.llamacpp_parallel < 1:
                    raise RuntimeError("context and parallel sequence settings must be positive")
                if self.model_context_capability is None and settings.max_context_length > 32768:
                    raise RuntimeError(
                        "the GGUF does not declare its trained context length; use a context of "
                        "32768 or less unless the model metadata is corrected"
                    )
                related = next(
                    (
                        entry
                        for entry in discover_gguf(model_path.parent)
                        if Path(entry.path) == model_path
                    ),
                    None,
                )
                if related is None and re.search(r"-\d{5}-of-\d{5}$", model_path.stem):
                    raise RuntimeError(
                        f"split GGUF shards are incomplete beside {model_path}; "
                        "restore every shard before starting the model"
                    )
                model_bytes = related.size_bytes if related else model_path.stat().st_size
                draft_model_bytes = 0
                if settings.llamacpp_draft_model:
                    draft_path = Path(settings.llamacpp_draft_model).expanduser()
                    if not draft_path.is_file():
                        raise RuntimeError(
                            f"configured speculative-decoding draft model is missing: {draft_path}"
                        )
                    draft_model_bytes = draft_path.stat().st_size
                try:
                    memory_estimate = estimate_runtime_memory(
                        model_bytes=model_bytes,
                        metadata=metadata,
                        context_tokens=settings.max_context_length,
                        parallel_sequences=settings.llamacpp_parallel,
                        batch_size=settings.llamacpp_batch_size,
                        ubatch_size=settings.llamacpp_ubatch_size,
                        cache_ram_bytes=settings.llamacpp_cache_ram_mib * 1024**2,
                        prompt_cache_enabled=settings.llamacpp_cache_prompt,
                        draft_model_bytes=draft_model_bytes,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"managed runtime memory configuration is invalid: {exc}"
                    ) from exc
                self.memory_estimate = memory_estimate
                estimate_bytes = memory_estimate.estimated_bytes
                if settings.model_memory_estimate_gb:
                    estimate_bytes = max(
                        estimate_bytes, int(settings.model_memory_estimate_gb * 1024**3)
                    )
                if estimate_bytes > memory_estimate.estimated_bytes:
                    components = dict(memory_estimate.components_bytes)
                    components["operator_supplied_estimate_floor"] = (
                        estimate_bytes - memory_estimate.estimated_bytes
                    )
                    memory_estimate = replace(
                        memory_estimate,
                        components_bytes=components,
                        estimated_bytes=estimate_bytes,
                    )
                    self.memory_estimate = memory_estimate
                estimate = estimate_bytes / 1024**3
                self.model_memory_estimate_gb = round(estimate, 2)
                require_memory_capacity(memory_estimate, estimate_bytes=estimate_bytes)
                log_path = Path(settings.llamacpp_log_file).expanduser()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                rotated = log_path.with_suffix(log_path.suffix + ".1")
                if (
                    log_path.exists()
                    and log_path.stat().st_size >= settings.llamacpp_log_max_mib * 1024**2
                ):
                    rotated.unlink(missing_ok=True)
                    log_path.replace(rotated)
                self._log = log_path.open("ab", buffering=0)
                self._log_path = log_path
                self._log.seek(0, 2)
                self._log_size = self._log.tell()
                self.process = await asyncio.create_subprocess_exec(
                    *self.command(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
                self.last_return_code = None
                self._attach_log_capture()
                await self._wait_healthy()
                self.state = RuntimeState.READY
                self._monitor = asyncio.create_task(self._watch(), name="llama-server-supervisor")
            except BaseException as exc:
                self.state = RuntimeState.FAILED
                self.last_error = str(exc)
                await self._stop_owned_child()
                raise

    async def _wait_healthy(self) -> None:
        deadline = asyncio.get_running_loop().time() + settings.llamacpp_startup_timeout_seconds
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._stopping:
                    raise RuntimeError("managed llama-server startup was cancelled by shutdown")
                if not self.process or self.process.returncode is not None:
                    raise RuntimeError("managed llama-server child exited before becoming healthy")
                try:
                    response = await client.get(f"{settings.llamacpp_base_url}/health")
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        raise RuntimeError(
            "managed llama-server did not become healthy within "
            f"{settings.llamacpp_startup_timeout_seconds:g} seconds"
        )

    def _attach_log_capture(self) -> None:
        if self.process and self.process.stdout:
            self._log_task = asyncio.create_task(
                self._capture_logs(self.process.stdout), name="llama-server-log-capture"
            )

    async def _capture_logs(self, stream: asyncio.StreamReader) -> None:
        max_bytes = settings.llamacpp_log_max_mib * 1024**2
        while chunk := await stream.read(64 * 1024):
            if self._log is None or self._log_path is None:
                continue
            if self._log_size + len(chunk) > max_bytes:
                self._log.close()
                rotated = self._log_path.with_suffix(self._log_path.suffix + ".1")
                rotated.unlink(missing_ok=True)
                self._log_path.replace(rotated)
                self._log = self._log_path.open("wb", buffering=0)
                self._log_size = 0
            self._log.write(chunk)
            self._log_size += len(chunk)

    async def _watch(self) -> None:
        while not self._stopping:
            process = self.process
            if process is None:
                return
            code = await process.wait()
            self.last_return_code = code
            if not self._stopping:
                self.state = RuntimeState.FAILED
            if self._log_task:
                await asyncio.gather(self._log_task, return_exceptions=True)
                self._log_task = None
            if self._stopping:
                return
            self.restart_count += 1
            self._recovery_attempts += 1
            logger.error("managed llama-server child exited with status %s; restarting", code)
            if (
                not settings.llamacpp_restart_enabled
                or self._recovery_attempts > settings.llamacpp_max_restarts
            ):
                self.state = RuntimeState.FAILED
                self.last_error = (
                    f"managed llama-server exited with status {code}; restart limit reached"
                )
                self.process = None
                return
            while not self._stopping:
                try:
                    async with self._lock:
                        if self._stopping:
                            return
                        self.state = RuntimeState.STARTING
                        self.process = await asyncio.create_subprocess_exec(
                            *self.command(),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                            start_new_session=True,
                        )
                        self.last_return_code = None
                        self._attach_log_capture()
                        await self._wait_healthy()
                        self.state = RuntimeState.READY
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.state = RuntimeState.FAILED
                    self.last_error = str(exc)
                    logger.exception("unable to restart managed llama-server")
                    async with self._lock:
                        await self._stop_owned_child(close_log=False)
                    self.restart_count += 1
                    self._recovery_attempts += 1
                    if self._recovery_attempts > settings.llamacpp_max_restarts:
                        self._stopping = True
                        return
                    if not self._stopping:
                        await asyncio.sleep(settings.llamacpp_restart_delay_seconds)

    async def _stop_owned_child(self, *, close_log: bool = True) -> None:
        process = self.process
        self.process = None
        if process and process.returncode is not None:
            self.last_return_code = process.returncode
        if process and process.returncode is None:
            # start_new_session=True gives this supervisor a private process
            # group. Signal only that owned group so helper descendants cannot
            # outlive the llama-server child.
            shutdown_deadline = (
                asyncio.get_running_loop().time() + settings.shutdown_timeout_seconds
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=max(0.01, shutdown_deadline - asyncio.get_running_loop().time()),
                )
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            else:
                # Allow any owned descendants the remaining graceful window,
                # then force-kill the private group if one is still present.
                group_alive = True
                while asyncio.get_running_loop().time() < shutdown_deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        group_alive = False
                        break
                    await asyncio.sleep(0.1)
                if group_alive:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if self._log_task:
            await asyncio.gather(self._log_task, return_exceptions=True)
            self._log_task = None
        if close_log and self._log:
            self._log.close()
            self._log = None

    async def stop(self) -> None:
        self._stopping = True
        async with self._lock:
            self.state = RuntimeState.STOPPING
            monitor, self._monitor = self._monitor, None
            if monitor and monitor is not asyncio.current_task():
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            await self._stop_owned_child()
            self.state = RuntimeState.STOPPED

    def status(self) -> dict[str, object]:
        process = self.process
        return {
            "state": self.state.value,
            "managed": settings.llamacpp_managed,
            "pid": process.pid if process and process.returncode is None else None,
            "owns_process": bool(process and process.returncode is None),
            "return_code": process.returncode if process else self.last_return_code,
            "restart_count": self.restart_count,
            "model_id": self.active_model_id,
            "model_path": self.active_model_path if settings.llamacpp_managed else None,
            "context_length": settings.max_context_length,
            "server_context_capacity": settings.max_context_length * settings.llamacpp_parallel,
            "model_context_capability": self.model_context_capability,
            "estimated_memory_gib": self.model_memory_estimate_gb,
            "memory_estimate": (
                {
                    "kind": "estimated",
                    "complete": self.memory_estimate.complete,
                    "estimated_bytes": self.memory_estimate.estimated_bytes,
                    "headroom_bytes": self.memory_estimate.headroom_bytes,
                    "capacity_bytes": self.memory_estimate.capacity_bytes,
                    "components_bytes": self.memory_estimate.components_bytes,
                    "notes": list(self.memory_estimate.notes),
                }
                if self.memory_estimate
                else None
            ),
            "observed_process_rss_bytes": _process_rss_bytes(process),
            "last_error": self.last_error,
            "supported_optional_flags": sorted(
                flag
                for flag in OPTIONAL_FLAGS
                if self.supported_flags and flag in self.supported_flags
            ),
        }

    def capabilities(self) -> dict[str, object]:
        supported = self.supported_flags
        return {
            "process_management": settings.llamacpp_managed,
            "model_discovery": True,
            "max_simultaneously_loaded_models": 1,
            "managed": settings.llamacpp_managed,
            "prompt_cache": None if supported is None else "--cache-prompt" in supported,
            "cache_ram_limit": None if supported is None else "--cache-ram" in supported,
            "kv_cache_type_k": None if supported is None else "--cache-type-k" in supported,
            "kv_cache_type_v": None if supported is None else "--cache-type-v" in supported,
            "flash_attention": None if supported is None else "--flash-attn" in supported,
            "parallel_slots": None if supported is None else "--parallel" in supported,
            "jinja": None if supported is None else "--jinja" in supported,
            "engine_metrics": None if supported is None else "--metrics" in supported,
            "mlock": None if supported is None else "--mlock" in supported,
            "gpu_layers": None if supported is None else "--n-gpu-layers" in supported,
            "cache_reuse": None if supported is None else "--cache-reuse" in supported,
            "reasoning_format": None if supported is None else "--reasoning-format" in supported,
            "draft_model": None if supported is None else "--draft-model" in supported,
            "draft_max_len": None if supported is None else "--draft-max-len" in supported,
            "draft_min_len": None if supported is None else "--draft-min-len" in supported,
            "draft_ctx_size": None if supported is None else "--draft-ctx-size" in supported,
            "embeddings": (
                settings.llamacpp_embeddings and (supported is None or "--embedding" in supported)
            ),
            "threads": settings.llamacpp_threads,
            "batch_threads": settings.llamacpp_batch_threads,
            "parallel": settings.llamacpp_parallel,
            "cache_telemetry": False,
            "engine_metrics_url": (
                f"http://{settings.llamacpp_host}:{settings.llamacpp_port}/metrics"
                if settings.llamacpp_metrics and settings.llamacpp_base_url
                else None
            ),
            "configured_cache": {
                "prompt_cache": settings.llamacpp_cache_prompt,
                "cache_ram_mib": settings.llamacpp_cache_ram_mib,
                "cache_reuse": settings.llamacpp_cache_reuse,
                "kv_cache_type_k": settings.llamacpp_cache_type_k or "f16",
                "kv_cache_type_v": settings.llamacpp_cache_type_v or "f16",
            },
        }


runtime = LlamaCppRuntime()


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def validate_runtime_tuning() -> None:
    """Warn about engine configurations that silently hurt throughput or safety.

    Startup stays permissive: an operator may deliberately run an unusual
    profile, but every known footgun is surfaced once in the gateway log.
    """
    if not settings.llamacpp_base_url:
        return
    if settings.llamacpp_managed and settings.llamacpp_host not in LOOPBACK_HOSTS:
        logger.warning(
            "managed llama-server is bound to %s without authentication; the OpenAI endpoint "
            "bypasses gateway auth, admission and queueing. Bind 127.0.0.1 or firewall the port",
            settings.llamacpp_host,
        )
    if settings.llamacpp_batch_size < settings.llamacpp_ubatch_size:
        logger.warning(
            "LLAMACPP_BATCH_SIZE (%s) is smaller than LLAMACPP_UBATCH_SIZE (%s); "
            "llama.cpp may refuse to start",
            settings.llamacpp_batch_size,
            settings.llamacpp_ubatch_size,
        )
    if settings.max_concurrent_requests > settings.llamacpp_parallel:
        logger.warning(
            "MAX_CONCURRENT_REQUESTS (%s) exceeds LLAMACPP_PARALLEL (%s); excess requests will "
            "queue inside llama-server and may hit gateway timeouts instead of a clean 429",
            settings.max_concurrent_requests,
            settings.llamacpp_parallel,
        )
    logical = psutil.cpu_count(logical=True) or 1
    if settings.llamacpp_threads > logical:
        logger.warning(
            "LLAMACPP_THREADS (%s) exceeds the %s logical CPUs; threading above the physical core "
            "count usually reduces tokens/s",
            settings.llamacpp_threads,
            logical,
        )
    if settings.llamacpp_parallel > 1 and settings.max_concurrent_requests <= 1:
        logger.warning(
            "LLAMACPP_PARALLEL is %s but MAX_CONCURRENT_REQUESTS is %s; the extra llama.cpp "
            "slots cannot be used and the KV cache is reserved for them",
            settings.llamacpp_parallel,
            settings.max_concurrent_requests,
        )
    quantized = [
        f"{name}={value}"
        for name, value in (
            ("LLAMACPP_CACHE_TYPE_K", settings.llamacpp_cache_type_k),
            ("LLAMACPP_CACHE_TYPE_V", settings.llamacpp_cache_type_v),
        )
        if value and value.lower() not in UNQUANTIZED_CACHE_TYPES
    ]
    if quantized and not settings.llamacpp_flash_attention:
        logger.warning(
            "quantized KV cache (%s) requires flash attention in llama.cpp; set "
            "LLAMACPP_FLASH_ATTENTION=true or keep an f16 cache, otherwise the engine may exit",
            ", ".join(quantized),
        )
    if settings.llamacpp_managed and not settings.llamacpp_jinja:
        logger.warning(
            "LLAMACPP_JINJA=false disables the model chat template: tool calls, reasoning "
            "extraction and role handling degrade compared with other OpenAI engines"
        )
    if settings.llamacpp_draft_model:
        supported = runtime.supported_flags
        if supported is not None and "--draft-model" not in supported:
            logger.warning(
                "LLAMACPP_DRAFT_MODEL=%r is set but this llama-server build lacks --draft-model; "
                "speculative decoding stays disabled",
                settings.llamacpp_draft_model,
            )


def require_memory_capacity(estimate: MemoryEstimate, *, estimate_bytes: int | None = None) -> None:
    required = estimate.estimated_bytes if estimate_bytes is None else estimate_bytes
    if required + estimate.headroom_bytes > estimate.capacity_bytes:
        raise RuntimeError(
            "estimated managed runtime memory plus headroom exceeds admission capacity: "
            f"{required / 1024**3:.2f} GiB estimated + "
            f"{estimate.headroom_bytes / 1024**3:.2f} GiB headroom > "
            f"{estimate.capacity_bytes / 1024**3:.2f} GiB capacity; "
            f"components={estimate.components_bytes}"
        )


def _process_rss_bytes(process: asyncio.subprocess.Process | None) -> int | None:
    if process is None or process.returncode is not None:
        return None
    try:
        return psutil.Process(process.pid).memory_info().rss
    except (OSError, psutil.Error):
        return None
