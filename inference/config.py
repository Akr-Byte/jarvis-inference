from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _physical_cores() -> int:
    """Best-effort physical core count; CPU-heavy defaults should not use SMT siblings."""
    try:
        import psutil

        return max(1, psutil.cpu_count(logical=False) or 8)
    except Exception:  # pragma: no cover - psutil is a hard dependency
        return 8


def _cpu_quota() -> int | None:
    """Return the effective CPU quota when the process is container-limited."""
    try:
        values = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()
        if len(values) == 2 and values[0] != "max":
            quota, period = int(values[0]), int(values[1])
            if period > 0:
                return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text(encoding="utf-8"))
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8"))
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return None


def _available_cores() -> int:
    """Physical cores bounded by cgroup quota and CPU affinity."""
    cores = _physical_cores()
    quota = _cpu_quota()
    if quota is not None:
        cores = min(cores, quota)
    try:
        cores = min(cores, len(os.sched_getaffinity(0)))
    except AttributeError:  # pragma: no cover - non-Linux
        pass
    return max(1, cores)


def _model_id_from_path(value: str) -> str:
    stem = Path(value).expanduser().stem
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def _default_model() -> str:
    configured = os.getenv("DEFAULT_MODEL", "").strip()
    if configured:
        return configured
    mapped = next(
        (
            entry.split("=", 1)[0].strip()
            for entry in os.getenv("LLAMACPP_MODEL_MAP", "").split(";")
            if "=" in entry
        ),
        "",
    )
    if mapped:
        return mapped
    alias = os.getenv("LLAMACPP_ALIAS", "").strip()
    if alias:
        return alias
    model_path = os.getenv("LLAMACPP_MODEL", "").strip()
    return _model_id_from_path(model_path) if model_path else "qwen3:1.7b"


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.getenv("INFERENCE_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("INFERENCE_PORT", 8080))
    api_key: str = field(default_factory=lambda: os.getenv("INFERENCE_API_KEY", "").strip())
    allow_insecure_no_auth: bool = field(
        default_factory=lambda: (
            os.getenv("INFERENCE_ALLOW_INSECURE_NO_AUTH", "false").strip().lower()
            in {"1", "true", "yes"}
        )
    )
    max_concurrent_requests: int = field(
        default_factory=lambda: max(1, _int("MAX_CONCURRENT_REQUESTS", 1))
    )
    # asyncio.Queue treats maxsize=0 as "unbounded" (GW-1), so clamp to 1+.
    max_queue_size: int = field(default_factory=lambda: max(1, _int("MAX_QUEUE_SIZE", 8)))
    max_request_body_mib: float = field(
        default_factory=lambda: max(0.05, _float("MAX_REQUEST_BODY_MIB", 8.0))
    )
    max_loaded_models: int = field(default_factory=lambda: max(1, _int("MAX_LOADED_MODELS", 1)))
    memory_budget_gb: float = field(
        default_factory=lambda: max(0.5, _float("MEMORY_BUDGET_GB", 10.0))
    )
    memory_headroom_gb: float = field(
        default_factory=lambda: max(1.0, _float("MEMORY_HEADROOM_GB", 1.0))
    )
    model_memory_estimate_gb: float = field(
        default_factory=lambda: max(0.0, _float("MODEL_MEMORY_ESTIMATE_GB", 0.0))
    )
    backend_http_timeout_seconds: float = field(
        default_factory=lambda: max(
            1.0,
            _float(
                "BACKEND_HTTP_TIMEOUT_SECONDS",
                float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600")),
            ),
        )
    )
    chat_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("CHAT_TIMEOUT_SECONDS", 300.0))
    )
    stream_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("STREAM_TIMEOUT_SECONDS", 300.0))
    )
    embedding_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("EMBEDDING_TIMEOUT_SECONDS", 30.0))
    )
    queue_timeout_seconds: float = field(
        default_factory=lambda: max(0.1, _float("QUEUE_TIMEOUT_SECONDS", 90.0))
    )
    shutdown_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("SHUTDOWN_TIMEOUT_SECONDS", 30.0))
    )
    model_refresh_seconds: float = field(
        default_factory=lambda: max(1.0, _float("MODEL_REFRESH_SECONDS", 15.0))
    )
    circuit_failure_threshold: int = field(
        default_factory=lambda: max(1, _int("CIRCUIT_FAILURE_THRESHOLD", 3))
    )
    circuit_recovery_seconds: float = field(
        default_factory=lambda: max(1.0, _float("CIRCUIT_RECOVERY_SECONDS", 15.0))
    )
    # A half-open trial request that exits without a verdict (client
    # disconnect, admission rejection, cancellation) must not wedge the
    # circuit: after this timeout a fresh trial is allowed (GW-15).
    circuit_trial_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("CIRCUIT_TRIAL_TIMEOUT_SECONDS", 60.0))
    )
    default_model: str = field(default_factory=_default_model)
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    ollama_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
    )
    # Explicit inference backend selector ("ollama" or "llamacpp"). Empty
    # keeps the legacy derivation: LLAMACPP_URL / LLAMACPP_MANAGED selects
    # llama.cpp, anything else falls back to Ollama.
    backend: str = field(default_factory=lambda: os.getenv("INFERENCE_BACKEND", "").strip().lower())
    llamacpp_url: str = field(default_factory=lambda: os.getenv("LLAMACPP_URL", "").rstrip("/"))
    llamacpp_managed: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_MANAGED", "false").strip().lower() in {"1", "true", "yes"}
        )
    )
    llamacpp_server: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_SERVER", "llama-server")
    )
    llamacpp_model: str = field(default_factory=lambda: os.getenv("LLAMACPP_MODEL", ""))
    llamacpp_model_map_raw: str = field(default_factory=lambda: os.getenv("LLAMACPP_MODEL_MAP", ""))
    model_directory: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_DIRECTORY", "~/models")).expanduser()
    )
    llamacpp_alias: str = field(default_factory=lambda: os.getenv("LLAMACPP_ALIAS", ""))
    llamacpp_host: str = field(default_factory=lambda: os.getenv("LLAMACPP_HOST", "127.0.0.1"))
    llamacpp_port: int = field(default_factory=lambda: _int("LLAMACPP_PORT", 8081))
    llamacpp_threads: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_THREADS", _available_cores()))
    )
    llamacpp_batch_threads: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_BATCH_THREADS", _available_cores()))
    )
    llamacpp_batch_size: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_BATCH_SIZE", 512))
    )
    llamacpp_ubatch_size: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_UBATCH_SIZE", 128))
    )
    llamacpp_parallel: int = field(default_factory=lambda: max(1, _int("LLAMACPP_PARALLEL", 1)))
    llamacpp_gpu_layers: int = field(default_factory=lambda: max(0, _int("LLAMACPP_GPU_LAYERS", 0)))
    llamacpp_flash_attention: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_FLASH_ATTENTION", "false").strip().lower() in {"1", "true", "yes"}
        )
    )
    # --jinja enables the model's own chat template (tool calling, correct
    # system/assistant roles, reasoning blocks). Without it llama-server uses
    # a generic template that silently degrades agent prompts.
    llamacpp_jinja: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_JINJA", "true").strip().lower() in {"1", "true", "yes"}
        )
    )
    # --metrics exposes llama-server's own Prometheus endpoint so operators can
    # inspect prompt-cache hits, slot utilisation and tokens/s independently.
    llamacpp_metrics: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_METRICS", "true").strip().lower() in {"1", "true", "yes"}
        )
    )
    # --mlock keeps model weights resident on CPU workstations; it is opt-in
    # because constrained hosts can be pushed into OOM by locked pages.
    llamacpp_mlock: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_MLOCK", "false").strip().lower() in {"1", "true", "yes"}
        )
    )
    # "auto" lets recent builds lift reasoning_content out of the message body.
    llamacpp_reasoning_format: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_REASONING_FORMAT", "").strip().lower()
    )
    # Optional speculative-decoding draft model GGUF path. When set and the
    # installed llama-server supports --draft-model, the engine runs a
    # draft-then-verify loop, the largest CPU-only throughput lever available.
    llamacpp_draft_model: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_DRAFT_MODEL", "").strip()
    )
    # Speculative-decoding sampling bounds; emitted only when a draft model is
    # configured. Lower bounds start drafting earlier; both must be positive.
    llamacpp_draft_max_len: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_DRAFT_MAX_LEN", 5))
    )
    llamacpp_draft_min_len: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_DRAFT_MIN_LEN", 5))
    )
    # Optional context size for the speculative-decoding draft model; 0 keeps
    # the engine default and only emits --draft-ctx-size when explicitly set.
    llamacpp_draft_ctx_size: int = field(
        default_factory=lambda: max(0, _int("LLAMACPP_DRAFT_CTX_SIZE", 0))
    )
    # Minimum chunk size (tokens) for prefix reuse between related prompts;
    # 0 disables the extra reuse pass.
    llamacpp_cache_reuse: int = field(
        default_factory=lambda: max(0, _int("LLAMACPP_CACHE_REUSE", 0))
    )
    llamacpp_embeddings: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_EMBEDDINGS", "false").strip().lower() in {"1", "true", "yes"}
        )
    )
    llamacpp_cache_prompt: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_CACHE_PROMPT", "true").strip().lower() in {"1", "true", "yes"}
        )
    )
    llamacpp_cache_ram_mib: int = field(
        default_factory=lambda: max(0, _int("LLAMACPP_CACHE_RAM_MIB", 0))
    )
    llamacpp_cache_type_k: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_CACHE_TYPE_K", "")
    )
    llamacpp_cache_type_v: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_CACHE_TYPE_V", "")
    )
    llamacpp_log_file: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_LOG_FILE", "logs/llama-server.log")
    )
    llamacpp_log_max_mib: int = field(
        default_factory=lambda: max(1, _int("LLAMACPP_LOG_MAX_MIB", 50))
    )
    llamacpp_restart_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("LLAMACPP_RESTART_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
        )
    )
    llamacpp_max_restarts: int = field(
        default_factory=lambda: max(0, _int("LLAMACPP_MAX_RESTARTS", 5))
    )
    llamacpp_restart_delay_seconds: float = field(
        default_factory=lambda: max(0.1, _float("LLAMACPP_RESTART_DELAY_SECONDS", 3.0))
    )
    llamacpp_startup_timeout_seconds: float = field(
        default_factory=lambda: max(1.0, _float("LLAMACPP_STARTUP_TIMEOUT_SECONDS", 300.0))
    )
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
    # Qwen3-family models think by default, which silently consumes the token
    # budget with reasoning. Default to explicit non-thinking unless a request
    # (or this env var) opts in.
    ollama_think_default: bool = field(
        default_factory=lambda: (
            os.getenv("OLLAMA_DEFAULT_THINK", "false").strip().lower() in {"1", "true", "yes"}
        )
    )
    max_context_length: int = field(
        default_factory=lambda: max(512, _int("MAX_CONTEXT_LENGTH", 16384))
    )
    max_output_tokens: int = field(default_factory=lambda: max(1, _int("MAX_OUTPUT_TOKENS", 2048)))
    max_embedding_batch_size: int = field(
        default_factory=lambda: max(1, _int("MAX_EMBEDDING_BATCH_SIZE", 64))
    )

    @property
    def request_timeout_seconds(self) -> float:
        """Backward-compatible alias for the backend transport timeout."""
        return self.backend_http_timeout_seconds

    @property
    def llamacpp_base_url(self) -> str:
        if self.llamacpp_url:
            return self.llamacpp_url
        if self.llamacpp_managed:
            return f"http://{self.llamacpp_host}:{self.llamacpp_port}"
        return ""

    @property
    def active_backend(self) -> str:
        """Active inference backend: "ollama" or "llama.cpp"."""
        if self.backend == "ollama":
            return "ollama"
        if self.backend == "llamacpp":
            return "llama.cpp"
        return "llama.cpp" if self.llamacpp_base_url else "ollama"

    @property
    def llamacpp_model_map(self) -> dict[str, str]:
        entries: dict[str, str] = {}
        for item in self.llamacpp_model_map_raw.split(";"):
            if "=" not in item:
                continue
            model_id, path = item.split("=", 1)
            if model_id.strip() and path.strip():
                entries[model_id.strip()] = path.strip()
        return entries

    @property
    def models(self) -> tuple[str, ...]:
        fallback = ",".join(self.llamacpp_model_map) or self.llamacpp_alias
        if not fallback and self.llamacpp_model:
            fallback = _model_id_from_path(self.llamacpp_model)
        fallback = fallback or "qwen3:1.7b,qwen3:4b"
        return _csv("MODEL_REGISTRY", fallback)

    @property
    def inference_models(self) -> tuple[str, ...]:
        if (self.llamacpp_managed or self.llamacpp_url) and not self.llamacpp_embeddings:
            return self.models
        return (*self.models, self.embedding_model)

    @property
    def ollama_models(self) -> tuple[str, ...]:
        return _csv("OLLAMA_MODELS", ",".join(self.inference_models))


settings = Settings()


def validate_security_config() -> None:
    """Fail startup closed unless auth is configured or insecure mode is explicit."""
    if not settings.api_key and not settings.allow_insecure_no_auth:
        raise RuntimeError(
            "INFERENCE_API_KEY is required. For isolated local development only, "
            "set INFERENCE_ALLOW_INSECURE_NO_AUTH=true explicitly."
        )
