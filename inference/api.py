from __future__ import annotations

import base64
import hmac
import json
import logging
import re
import struct
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask

from inference.config import settings
from inference.errors import InferenceError
from inference.llamacpp_runtime import runtime
from inference.metrics import render
from inference.model_registry import model_details, registry_status
from inference.models import model_manager
from inference.resources import status as resource_status
from inference.scheduler import StreamLease, scheduler
from inference.schemas import ChatCompletionRequest, EmbeddingRequest, ModelInfo

router = APIRouter()

logger = logging.getLogger("jarvis.api")

# Client-supplied request ids are echoed into response headers and payload id
# fields; only bounded alphanumeric ids are accepted (GW-13).
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:@/-]{1,128}")

# Parameters tolerated by the request schema (extra="allow") but not
# implemented by this gateway. They are rejected with an explicit
# OpenAI-compatible 400 instead of being silently dropped (GW-11).
UNSUPPORTED_CHAT_PARAMS = frozenset(
    {
        "n",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "presence_penalty",
        "frequency_penalty",
    }
)


def _error_headers(request_id: str, exc: InferenceError | None = None) -> dict[str, str]:
    headers = {"X-Request-ID": request_id}
    if exc is not None:
        if exc.code == "QUEUE_TIMEOUT":
            headers["Retry-After"] = "5"
        elif exc.code == "QUEUE_FULL":
            headers["Retry-After"] = "2"
    return headers


def _error_response(exc: InferenceError, request_id: str) -> JSONResponse:
    """OpenAI-compatible `{"error": {...}}` envelope (GW-7).

    FastAPI's default `{"detail": ...}` envelope breaks OpenAI-SDK error
    handling, so InferenceError paths return this shape directly.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": request_id,
            }
        },
        headers=_error_headers(request_id, exc),
    )


def _encode_embedding(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "float":
        return vector
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return base64.b64encode(packed).decode("ascii")


def _auth(value: str | None) -> None:
    if not settings.api_key:
        if settings.allow_insecure_no_auth:
            return
        raise HTTPException(503, "Inference API authentication is not configured")
    if not hmac.compare_digest(value or "", settings.api_key):
        raise HTTPException(401, "Invalid inference API key")


def _bearer(value: str | None) -> str | None:
    return value.removeprefix("Bearer ").strip() if value else None


def _request_id(request: Request) -> str:
    """Echo a client-supplied X-Request-ID only when it is safe (GW-13).

    The value is reused in response headers, JSON `id` fields and tool-call
    ids, so it must be bounded and restricted to header-safe characters;
    anything else is replaced with a generated id.
    """
    candidate = (request.headers.get("X-Request-ID") or "").strip()
    if candidate and len(candidate) <= 128 and _SAFE_REQUEST_ID.fullmatch(candidate):
        return candidate
    return f"chatcmpl-{uuid.uuid4().hex}"


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "jarvis-inference"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    readiness = await scheduler.readiness()
    if isinstance(readiness, bool):
        readiness = {"ready": readiness}
    body = {
        "status": "ready" if readiness["ready"] else "not_ready",
        "backend": scheduler.backend.name,
        "models": list(settings.models),
        **readiness,
    }
    return JSONResponse(content=body, status_code=200 if readiness["ready"] else 503)


@router.post("/v1/embeddings")
async def embeddings(
    payload: EmbeddingRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _auth(_bearer(authorization))
    request_id = _request_id(request)
    model = payload.model or settings.embedding_model
    inputs = [payload.input] if isinstance(payload.input, str) else payload.input
    try:
        vectors = await scheduler.embeddings(model, inputs, request_id=request_id)
    except InferenceError as exc:
        return _error_response(exc, request_id)
    # Embedding backends do not report token counts; report a conservative
    # estimate instead of a dishonest zero (GW-11).
    estimated_tokens = max(1, (sum(len(str(item)) for item in inputs) + 2 * len(inputs)) // 3)
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": _encode_embedding(vector, payload.encoding_format or "float"),
                    "index": index,
                }
                for index, vector in enumerate(vectors)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": estimated_tokens,
                "total_tokens": estimated_tokens,
                "prompt_tokens_estimated": True,
            },
        },
        headers={"X-Request-ID": request_id},
    )


@router.get("/v1/models", response_model=dict)
async def models(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _auth(_bearer(authorization))
    return {
        "object": "list",
        "data": [ModelInfo(id=model).model_dump() for model in settings.models],
    }


@router.get("/v1/capabilities", response_model=dict)
async def capabilities(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    """Advertise the inference protocol and model capabilities."""
    _auth(_bearer(authorization))
    chat_models = list(settings.models)
    models = []
    for model in chat_models:
        models.append({"id": model, "capabilities": ["chat", "streaming"]})
    embeddings_enabled = scheduler.backend.name == "ollama" or (
        settings.llamacpp_embeddings
        and (not settings.llamacpp_managed or runtime.capabilities()["embeddings"] is True)
    )
    if embeddings_enabled:
        models.append({"id": settings.embedding_model, "capabilities": ["embeddings"]})
    features = ["chat", "streaming", "model_catalog", "request_ids"]
    if embeddings_enabled:
        features.append("embeddings")
    return {
        "protocol": {"current": 1, "min_client": 1, "max_client": 1},
        "service": "jarvis-inference",
        "features": features,
        "limitations": [
            "tool_choice is tolerated but not enforced by the backend "
            "(only tool_choice=none is applied, by stripping the tools)"
        ],
        "models": models,
        "limits": {
            "max_concurrent_requests": settings.max_concurrent_requests,
            "max_queue_size": settings.max_queue_size,
            "queue_timeout_seconds": settings.queue_timeout_seconds,
            "chat_timeout_seconds": settings.chat_timeout_seconds,
            "stream_timeout_seconds": settings.stream_timeout_seconds,
            "embedding_timeout_seconds": settings.embedding_timeout_seconds,
        },
    }


@router.get("/v1/inference/status")
async def inference_status(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _auth(_bearer(authorization))
    return {
        "runtime": scheduler.status(),
        "engine_runtime": runtime.status(),
        "model_registry": registry_status(),
        "resources": resource_status(),
    }


@router.get("/v1/runtime/status")
async def runtime_status(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _auth(_bearer(authorization))
    return {"runtime": runtime.status(), "scheduler": scheduler.status()}


@router.get("/v1/runtime/capabilities")
async def runtime_capabilities(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _auth(_bearer(authorization))
    return runtime.capabilities()


@router.get("/v1/model-registry")
async def model_registry_status(
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    _auth(_bearer(authorization))
    return registry_status()


@router.get("/v1/models/{model_id}")
async def model_metadata(model_id: str, authorization: str | None = Header(default=None)) -> object:
    _auth(_bearer(authorization))
    details = model_details(model_id)
    if details is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "UNKNOWN_MODEL", "message": f"Unknown model '{model_id}'"}},
        )
    return details


@router.get("/v1/models/{model_id}/status")
async def model_runtime_status(
    model_id: str, authorization: str | None = Header(default=None)
) -> object:
    _auth(_bearer(authorization))
    details = model_details(model_id)
    if details is None:
        return JSONResponse(status_code=404, content={"error": {"code": "UNKNOWN_MODEL"}})
    return {
        key: details[key]
        for key in ("id", "registered", "available", "loaded", "accepting_requests", "enabled")
    }


async def _model_lifecycle(model_id: str, load: bool) -> object:
    details = model_details(model_id)
    if details is None or not details["registered"]:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "UNKNOWN_MODEL",
                    "message": f"Model '{model_id}' is not registered",
                }
            },
        )
    if not settings.llamacpp_managed:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "LIFECYCLE_UNSUPPORTED",
                    "message": "Model lifecycle control requires managed llama.cpp mode",
                }
            },
        )
    configured_ids = set(settings.llamacpp_model_map)
    default_id = settings.llamacpp_alias or Path(settings.llamacpp_model).expanduser().stem
    default_id = re.sub(r"-\d{5}-of-\d{5}$", "", default_id)
    if model_id != default_id and model_id not in configured_ids:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "MODEL_PROFILE_UNAVAILABLE",
                    "message": "This model has no configured managed runtime profile",
                }
            },
        )
    if not scheduler.begin_runtime_maintenance():
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "MODEL_BUSY",
                    "message": "Wait for active and queued requests to finish before changing runtime state",
                }
            },
        )
    if load:
        # Pre-flight the target profile before unloading the working model: a
        # missing or unreadable GGUF must not cause avoidable downtime.
        target = settings.llamacpp_model_map.get(model_id)
        if target is None and model_id == default_id:
            target = settings.llamacpp_model
        target_path = Path(target).expanduser() if target else None
        if target_path is None or not target_path.is_file():
            scheduler.end_runtime_maintenance()
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "MODEL_FILE_MISSING",
                        "message": (
                            f"The configured GGUF for '{model_id}' is missing or not a file; "
                            "the current model was left loaded"
                        ),
                    }
                },
            )
    try:
        if load:
            current_id = runtime.active_model_id
            if runtime.process and runtime.process.returncode is None and current_id != model_id:
                await runtime.stop()
                if current_id:
                    model_manager.record_unloaded(current_id)
            await runtime.start(model_id)
            await model_manager.refresh(scheduler.backend, force=True)
            return {"id": model_id, "loaded": True, "runtime": runtime.status()}
        if runtime.active_model_id != model_id:
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "MODEL_NOT_LOADED",
                        "message": "The requested model is not the active runtime model",
                    }
                },
            )
        await runtime.stop()
        model_manager.record_unloaded(model_id)
        return {"id": model_id, "loaded": False, "runtime": runtime.status()}
    except (RuntimeError, OSError) as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "RUNTIME_LIFECYCLE_FAILED", "message": str(exc)}},
        )
    finally:
        scheduler.end_runtime_maintenance()


@router.post("/v1/models/{model_id}/load")
async def load_model(model_id: str, authorization: str | None = Header(default=None)) -> object:
    _auth(_bearer(authorization))
    return await _model_lifecycle(model_id, load=True)


@router.post("/v1/models/{model_id}/unload")
async def unload_model(model_id: str, authorization: str | None = Header(default=None)) -> object:
    _auth(_bearer(authorization))
    return await _model_lifecycle(model_id, load=False)


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(
    authorization: str | None = Header(default=None),
) -> PlainTextResponse:
    _auth(_bearer(authorization))
    return PlainTextResponse(render().decode())


@router.post("/v1/chat/completions")
async def chat(
    payload: ChatCompletionRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> object:
    _auth(_bearer(authorization))
    request_id = _request_id(request)
    body = payload.model_dump(exclude_none=True)
    body["model"] = body.get("model") or settings.default_model
    try:
        model_manager.validate_chat(str(body["model"]))
    except InferenceError as exc:
        return _error_response(exc, request_id)
    # Honest rejection for tolerated-but-unimplemented parameters (GW-11):
    # silently dropping them makes clients believe their settings took effect.
    unsupported = sorted(set(body).intersection(UNSUPPORTED_CHAT_PARAMS))
    if unsupported:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "UNSUPPORTED_PARAMETER",
                    "message": (
                        "parameters not supported by this gateway: " + ", ".join(unsupported)
                    ),
                    "retryable": False,
                    "request_id": request_id,
                }
            },
            headers=_error_headers(request_id),
        )
    # tool_choice honesty (GW-11): the backend ignores tool_choice entirely.
    # The only enforceable value is "none", which the gateway applies itself
    # by stripping the tools; "auto" is the backend default and other values
    # are tolerated but NOT enforced (advertised via /v1/capabilities).
    tool_choice = body.get("tool_choice")
    if tool_choice == "none":
        body.pop("tools", None)
        body.pop("tool_choice", None)
    if payload.stream:
        try:
            lease = await scheduler.prepare_stream(body, request_id=request_id)
        except InferenceError as exc:
            return _error_response(exc, request_id)
        return StreamingResponse(
            _stream(body, request_id, lease),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
            background=BackgroundTask(scheduler.release_stream, lease),
        )
    try:
        result = await scheduler.chat(body, request_id=request_id)
    except InferenceError as exc:
        return _error_response(exc, request_id)
    result.pop("_inference", None)
    result.setdefault("id", request_id)
    result.setdefault("created", int(time.time()))
    result.setdefault("object", "chat.completion")
    result.setdefault("choices", [])
    result.setdefault("usage", {})
    return JSONResponse(content=result, headers={"X-Request-ID": request_id})


async def _stream(
    body: dict[str, object], request_id: str, lease: StreamLease
) -> AsyncIterator[str]:
    model = str(body["model"])
    model_manager.validate_chat(model)
    backend = scheduler.backend
    openai_done_sent = False
    # OpenAI streaming semantics (GW-4a): when the client asks for usage via
    # stream_options.include_usage, the usage object ships in a final
    # usage-only chunk with empty choices — not inside the content chunk.
    stream_options = body.get("stream_options")
    include_usage = bool(isinstance(stream_options, dict) and stream_options.get("include_usage"))
    try:
        stream = await scheduler.stream(body, request_id=request_id, lease=lease)
        async for line in stream:
            if backend.name == "ollama":
                try:
                    data = backend.sse_data(line)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    # A malformed backend line must not abort an otherwise
                    # healthy stream; skip it and keep reading (GW-4a).
                    logger.warning(
                        "skipping malformed backend SSE line (request %s): %s",
                        request_id,
                        exc,
                    )
                    continue
                message = data.get("message") or {}
                delta: dict[str, object] = {"content": message.get("content", "")}
                if message.get("role"):
                    delta["role"] = message["role"]
                if message.get("thinking"):
                    delta["reasoning_content"] = message["thinking"]
                if message.get("tool_calls"):
                    # Ollama streams complete tool_calls objects; OpenAI-SDK
                    # stream parsers expect indexed deltas (GW-8).
                    delta["tool_calls"] = [
                        {
                            "index": position,
                            "id": f"{request_id}-call-{position}",
                            "type": "function",
                            "function": {
                                "name": str((call.get("function") or {}).get("name") or ""),
                                "arguments": (
                                    json.dumps(
                                        (call.get("function") or {}).get("arguments", {}),
                                        ensure_ascii=False,
                                    )
                                    if not isinstance(
                                        (call.get("function") or {}).get("arguments"),
                                        str,
                                    )
                                    else (call["function"]["arguments"] or "")
                                ),
                            },
                        }
                        for position, call in enumerate(message["tool_calls"])
                        if isinstance(call, dict)
                    ]
                finish_reason: str | None = None
                usage = None
                if data.get("done"):
                    # Map Ollama's done_reason exactly like the non-streaming
                    # response path, so tool calls finish as tool_calls.
                    finish_reason = (
                        "tool_calls"
                        if message.get("tool_calls")
                        else data.get("done_reason", "stop")
                    )
                    if data.get("eval_count") is not None:
                        usage = {
                            "prompt_tokens": int(data.get("prompt_eval_count") or 0),
                            "completion_tokens": int(data.get("eval_count") or 0),
                            "total_tokens": int(data.get("prompt_eval_count") or 0)
                            + int(data.get("eval_count") or 0),
                        }
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                if usage is not None and not include_usage:
                    chunk["usage"] = usage
                yield f"data: {json.dumps(chunk)}\n\n"
                if usage is not None and include_usage:
                    usage_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
            else:
                # llama.cpp streams OpenAI-format chunks verbatim, including
                # its own terminal `data: [DONE]` sentinel; only append the
                # gateway sentinel when the backend did not send one (GW-21).
                stripped = line.strip()
                if stripped == "[DONE]":
                    openai_done_sent = True
                    yield f"data: {line}\n\n"
                    continue
                if not include_usage and '"usage"' in stripped:
                    # The backend is always asked for a usage chunk so the
                    # gateway can meter tokens (see LlamaCppBackend.stream).
                    # Clients that did not set stream_options.include_usage
                    # must not receive the usage-only chunk (OpenAI contract).
                    try:
                        event = json.loads(stripped)
                    except (TypeError, ValueError):
                        event = None
                    if isinstance(event, dict) and event.get("usage") and not event.get("choices"):
                        continue
                yield f"data: {line}\n\n"
    except InferenceError as exc:
        # Once SSE headers are committed, communicate backend failures as an
        # OpenAI-compatible error event rather than abruptly truncating the stream.
        error_event = {
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "request_id": request_id,
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        yield "data: [DONE]\n\n"
        return
    if not openai_done_sent:
        yield "data: [DONE]\n\n"
