"""Lifecycle and capability management for an owned llama-server child."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from enum import StrEnum
from pathlib import Path

import httpx

from inference.config import settings
from inference.gguf import kv_bytes_per_token, read_metadata
from inference.model_registry import discover_gguf
from inference.resources import memory_available_gb

logger = logging.getLogger(__name__)

OPTIONAL_FLAGS = {
    "--alias": "alias",
    "--cache-prompt": "prompt_cache",
    "--cache-ram": "cache_ram_limit",
    "--cache-type-k": "kv_cache_type_k",
    "--cache-type-v": "kv_cache_type_v",
    "--flash-attn": "flash_attention",
    "--parallel": "parallel_slots",
    "--threads-batch": "batch_threads",
    "--embedding": "embeddings",
}


class RuntimeState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


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
        self.restart_count = 0
        self._recovery_attempts = 0
        self.supported_flags: set[str] | None = None
        self._capability_probe_error: str | None = None
        self.model_memory_estimate_gb: float | None = None
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
            str(settings.max_context_length),
            "--threads",
            str(settings.llamacpp_threads),
            "--batch-size",
            str(settings.llamacpp_batch_size),
            "--ubatch-size",
            str(settings.llamacpp_ubatch_size),
        ]
        optional: list[tuple[str, str | None, str]] = [
            ("--alias", self.active_model_id or settings.llamacpp_alias or None, "alias"),
            ("--cache-prompt", None, "prompt_cache"),
            (
                "--cache-ram",
                str(settings.llamacpp_cache_ram_mib) if settings.llamacpp_cache_ram_mib else None,
                "cache_ram_limit",
            ),
            ("--cache-type-k", settings.llamacpp_cache_type_k or None, "kv_cache_type_k"),
            ("--cache-type-v", settings.llamacpp_cache_type_v or None, "kv_cache_type_v"),
            (
                "--flash-attn",
                "auto" if settings.llamacpp_flash_attention else None,
                "flash_attention",
            ),
            ("--parallel", str(settings.llamacpp_parallel), "parallel_slots"),
            ("--threads-batch", str(settings.llamacpp_batch_threads), "batch_threads"),
            ("--embedding", None, "embeddings"),
        ]
        for flag, value, capability in optional:
            enabled = (
                value is not None
                or (flag == "--cache-prompt" and settings.llamacpp_cache_prompt)
                or (flag == "--embedding" and settings.llamacpp_embeddings)
            )
            if not enabled:
                continue
            if self.supported_flags is not None and flag not in self.supported_flags:
                logger.warning(
                    "llama-server does not support configured optional feature %s", capability
                )
                continue
            cmd.append(flag)
            if value is not None:
                cmd.append(value)
        return cmd

    async def start(self, model_id: str | None = None) -> None:
        if not settings.llamacpp_managed:
            return
        async with self._lock:
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
                self.model_context_capability = int(trained_context) if trained_context else None
                if (
                    self.model_context_capability
                    and settings.max_context_length > self.model_context_capability
                ):
                    raise RuntimeError(
                        f"configured context {settings.max_context_length} exceeds the model's "
                        f"trained context {self.model_context_capability}"
                    )
                kv_per_token = kv_bytes_per_token(
                    metadata,
                    settings.llamacpp_cache_type_k or "f16",
                    settings.llamacpp_cache_type_v or "f16",
                )
                if kv_per_token is None and settings.max_context_length > 32768:
                    raise RuntimeError(
                        "cannot estimate KV memory for this GGUF at context lengths above 32768; "
                        "set supported model metadata or use a smaller MAX_CONTEXT_LENGTH"
                    )
                if trained_context is None and settings.max_context_length > 32768:
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
                kv_bytes = (kv_per_token or 0) * settings.max_context_length
                estimate = max(
                    (model_bytes * 1.10 + kv_bytes + settings.llamacpp_cache_ram_mib * 1024**2)
                    / 1024**3,
                    settings.model_memory_estimate_gb,
                )
                self.model_memory_estimate_gb = round(estimate, 2)
                capacity = min(memory_available_gb(), settings.memory_budget_gb)
                if estimate + settings.memory_headroom_gb > capacity:
                    raise RuntimeError(
                        f"estimated model/runtime memory {estimate:.1f} GiB plus configured "
                        f"headroom {settings.memory_headroom_gb:.1f} GiB exceeds available "
                        f"inference budget {capacity:.1f} GiB; lower context or set a suitable "
                        "MEMORY_BUDGET_GB after checking host capacity"
                    )
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
                self._attach_log_capture()
                await self._wait_healthy()
                self.state = RuntimeState.READY
                self._stopping = False
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
                    self.state = RuntimeState.STARTING
                    self.process = await asyncio.create_subprocess_exec(
                        *self.command(),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        start_new_session=True,
                    )
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
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=settings.shutdown_timeout_seconds)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self._log_task:
            await asyncio.gather(self._log_task, return_exceptions=True)
            self._log_task = None
        if close_log and self._log:
            self._log.close()
            self._log = None

    async def stop(self) -> None:
        self.state = RuntimeState.STOPPING
        self._stopping = True
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
            "return_code": process.returncode if process else None,
            "restart_count": self.restart_count,
            "model_id": self.active_model_id,
            "model_path": self.active_model_path if settings.llamacpp_managed else None,
            "context_length": settings.max_context_length,
            "model_context_capability": self.model_context_capability,
            "estimated_memory_gib": self.model_memory_estimate_gb,
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
            "embeddings": (
                settings.llamacpp_embeddings and (supported is None or "--embedding" in supported)
            ),
            "cache_telemetry": False,
            "configured_cache": {
                "prompt_cache": settings.llamacpp_cache_prompt,
                "cache_ram_mib": settings.llamacpp_cache_ram_mib,
                "kv_cache_type_k": settings.llamacpp_cache_type_k or "f16",
                "kv_cache_type_v": settings.llamacpp_cache_type_v or "f16",
            },
        }


runtime = LlamaCppRuntime()
