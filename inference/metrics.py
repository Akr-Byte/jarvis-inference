from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

from inference.models import model_manager
from inference.resources import status as resource_status

REQUESTS = Counter("inference_requests_total", "Inference requests.", ["model", "status"])
LATENCY = Histogram("inference_request_duration_seconds", "Inference request duration.", ["model"])
QUEUE_WAIT = Histogram(
    "inference_queue_wait_seconds",
    "Time spent waiting for an inference slot.",
    ["model"],
)
TTFT = Histogram(
    "inference_time_to_first_token_seconds",
    "Time from backend start to first streamed token.",
    ["model"],
)
GENERATION_DURATION = Histogram(
    "inference_generation_duration_seconds",
    "Backend generation duration in seconds.",
    ["model"],
)
TOKENS = Counter("inference_tokens_total", "Inference tokens.", ["model", "kind"])
TOKENS_PER_SECOND = Histogram(
    "inference_tokens_per_second", "Observed output token generation rate.", ["model"]
)
MODEL_LOAD = Histogram("inference_model_load_seconds", "Observed model load duration.", ["model"])
QUEUE = Gauge("inference_queue_depth", "Requests waiting for an inference slot.")
ACTIVE_REQUESTS = Gauge("inference_active_requests", "Currently executing inference requests.")
ACTIVE_REQUEST_AGE = Gauge(
    "inference_active_request_age_seconds",
    "Age of the oldest active inference request.",
)
QUEUE_TIMEOUTS = Counter(
    "inference_queue_timeouts_total",
    "Requests rejected after waiting too long for an inference slot.",
)
CANCELLED_REQUESTS = Counter(
    "inference_cancelled_requests_total",
    "Inference requests cancelled by the caller.",
    ["kind"],
)
RESOURCE_REJECTIONS = Counter(
    "inference_resource_rejections_total", "Resource-pressure rejections."
)
BACKEND_ERRORS = Counter(
    "inference_backend_errors_total",
    "Backend failures and timeouts.",
    ["backend", "code"],
)
STALE_REQUESTS = Counter(
    "inference_stale_requests_total",
    "Abandoned requests whose slots were reclaimed by the sweeper.",
)
CIRCUIT_STATE = Gauge(
    "inference_circuit_open",
    "Whether the backend circuit is open (1) or closed.",
    ["backend"],
)
ACTIVE_MODEL = Gauge("inference_active_model", "Whether the model is active.", ["model"])
MODEL_WARM = Gauge("inference_model_warm", "Whether the model is currently warm.", ["model"])
HOST_MEMORY_AVAILABLE = Gauge(
    "inference_host_memory_available_gib", "Available host memory in GiB."
)
CONTAINER_MEMORY_AVAILABLE = Gauge(
    "inference_container_memory_available_gib",
    "Available inference-container memory in GiB.",
)
ADMISSION_MEMORY_AVAILABLE = Gauge(
    "inference_admission_memory_available_gib",
    "Memory available to model admission in GiB.",
)
MEMORY_BUDGET = Gauge(
    "inference_memory_budget_gib", "Configured model admission memory budget in GiB."
)
CPU_PERCENT = Gauge("inference_host_cpu_percent", "Host CPU utilization percentage.")
RUNTIME_STATE = Gauge("inference_runtime_state", "Managed engine lifecycle state.", ["state"])
RUNTIME_RESTARTS = Counter("inference_runtime_restarts_total", "Managed runtime restarts.")
RUNTIME_MEMORY_ESTIMATE = Gauge(
    "inference_runtime_estimated_memory_gib",
    "Estimated managed runtime memory in GiB; not observed process consumption.",
)
CONTEXT_USAGE = Histogram(
    "inference_context_usage_ratio",
    "Estimated or tokenizer-counted prompt plus output budget / context window.",
    ["model"],
)
_reported_runtime_restarts = 0


def render() -> bytes:
    global _reported_runtime_restarts
    resources = resource_status()
    HOST_MEMORY_AVAILABLE.set(float(resources["host_memory_available_gb"]))
    CONTAINER_MEMORY_AVAILABLE.set(float(resources["container_memory_available_gb"]))
    ADMISSION_MEMORY_AVAILABLE.set(float(resources["admission_memory_available_gb"]))
    MEMORY_BUDGET.set(float(resources["memory_budget_gb"]))
    CPU_PERCENT.set(float(resources["cpu_percent"]))
    from inference.llamacpp_runtime import RuntimeState, runtime

    runtime_status = runtime.status()
    for state in RuntimeState:
        RUNTIME_STATE.labels(state=state.value).set(
            1 if runtime_status["state"] == state.value else 0
        )
    restart_delta = int(runtime_status["restart_count"]) - _reported_runtime_restarts
    if restart_delta > 0:
        RUNTIME_RESTARTS.inc(restart_delta)
        _reported_runtime_restarts += restart_delta
    if runtime_status["estimated_memory_gib"] is not None:
        RUNTIME_MEMORY_ESTIMATE.set(float(runtime_status["estimated_memory_gib"]))
    state = model_manager.status()
    warm = set(state["warm_models"])
    for model in state["configured_models"]:
        MODEL_WARM.labels(model=model).set(1 if model in warm else 0)
    return generate_latest()
