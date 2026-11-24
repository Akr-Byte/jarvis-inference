from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from inference.backends.llamacpp import LlamaCppBackend
from inference.backends.ollama import OllamaBackend
from inference.circuit import CircuitBreaker, CircuitOpenError
from inference.config import settings
from inference.errors import InferenceError
from inference.metrics import (
    ACTIVE_MODEL,
    ACTIVE_REQUEST_AGE,
    ACTIVE_REQUESTS,
    BACKEND_ERRORS,
    CANCELLED_REQUESTS,
    CIRCUIT_STATE,
    CONTEXT_USAGE,
    GENERATION_DURATION,
    LATENCY,
    MODEL_LOAD,
    QUEUE,
    QUEUE_TIMEOUTS,
    QUEUE_WAIT,
    REQUESTS,
    RESOURCE_REJECTIONS,
    STALE_REQUESTS,
    TOKENS,
    TOKENS_PER_SECOND,
    TTFT,
)
from inference.models import model_manager
from inference.resources import admit_request


class StreamLine(str):
    """A raw SSE line carrying its decoded JSON event for downstream reuse."""

    event: dict[str, Any] | None

    def __new__(cls, raw: str, event: dict[str, Any] | None) -> StreamLine:
        obj = super().__new__(cls, raw)
        obj.event = event
        return obj


@dataclass
class StreamLease:
    """An admitted streaming request whose slot must be released exactly once."""

    model: str
    request_id: str
    released: bool = False
    payload: dict[str, Any] | None = None
    task: asyncio.Task[Any] | None = None

    def cancel(self) -> None:
        """Cancel the task driving this stream, if it is still running."""
        if self.task is not None and not self.task.done():
            self.task.cancel()


# Keys that do not belong to the rendered prompt. Kept in one place so the
# heuristic estimate and the exact tokenizer count agree on the payload.
_NON_PROMPT_KEYS = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "num_ctx",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "presence_penalty",
        "frequency_penalty",
    }
)
# Below this fraction of the context window an exact tokenizer count cannot
# change admission, so the extra engine round-trip is skipped.
TOKENIZER_ADMISSION_RATIO = 0.8


def _context_limits(payload: dict[str, Any]) -> tuple[int, int]:
    """Return the effective context window and the reserved output budget."""
    requested_context = payload.get("num_ctx")
    if requested_context is not None and int(requested_context) > settings.max_context_length:
        raise InferenceError(
            "CONTEXT_LIMIT_EXCEEDED",
            f"num_ctx cannot exceed the configured limit of {settings.max_context_length}",
            False,
            400,
        )
    effective = (
        int(requested_context) if requested_context is not None else settings.max_context_length
    )
    requested_output = payload.get("max_tokens")
    if requested_output is None:
        requested_output = payload.get("max_completion_tokens")
    output_budget = (
        int(requested_output) if requested_output is not None else settings.max_output_tokens
    )
    if output_budget > settings.max_output_tokens:
        raise InferenceError(
            "OUTPUT_LIMIT_EXCEEDED",
            f"max_tokens cannot exceed the configured limit of {settings.max_output_tokens}",
            False,
            400,
        )
    return effective, output_budget


def _prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in _NON_PROMPT_KEYS}


def _compact_payload(
    payload: dict[str, Any], effective: int, output_budget: int
) -> dict[str, Any] | None:
    """Return a suffix-only candidate that fits the heuristic budget, if one exists."""
    messages = payload.get("messages", [])
    if len(messages) <= 1:
        return None
    system = [msg for msg in messages if msg.get("role") == "system"]
    conversation = [msg for msg in messages if msg.get("role") != "system"]
    running = sum(estimate_prompt_tokens([msg]) for msg in system)
    newest_first: list[dict[str, Any]] = []
    for msg in reversed(conversation):
        msg_tokens = estimate_prompt_tokens([msg])
        if running + msg_tokens + output_budget > effective:
            break
        newest_first.append(msg)
        running += msg_tokens
    compacted_messages = system + list(reversed(newest_first))
    if len(compacted_messages) >= len(messages):
        return None
    compacted = dict(payload)
    compacted["messages"] = compacted_messages
    compacted_tokens = estimate_prompt_tokens(
        _prompt_payload(compacted) or compacted_messages
    )
    if compacted_tokens + output_budget > effective:
        return None
    return compacted


def _check_context_window(
    payload: dict[str, Any],
    actual_prompt_tokens: int | None = None,
    *,
    estimated_prompt_tokens: int | None = None,
    allow_compaction: bool = True,
) -> dict[str, Any]:
    """Return a payload that fits the effective context window (GW-6).

    Oversized multi-message chats are compacted by dropping the oldest
    non-system messages; the returned payload is the one callers must forward
    to the backend. A single oversized message cannot be compacted and is
    rejected. Set ``allow_compaction=False`` when an exact tokenizer already
    rejected the original prompt so a heuristic cannot re-admit it.
    """
    effective, output_budget = _context_limits(payload)
    estimated = (
        estimated_prompt_tokens
        if estimated_prompt_tokens is not None
        else estimate_prompt_tokens(_prompt_payload(payload) or payload.get("messages", []))
    )
    prompt_tokens = actual_prompt_tokens if actual_prompt_tokens is not None else estimated
    model_label = str(payload.get("model") or settings.default_model)
    CONTEXT_USAGE.labels(model=model_label).observe((prompt_tokens + output_budget) / effective)
    if prompt_tokens + output_budget <= effective:
        return payload

    if allow_compaction:
        compacted = _compact_payload(payload, effective, output_budget)
        if compacted is not None:
            compacted_tokens = estimate_prompt_tokens(
                _prompt_payload(compacted) or compacted.get("messages", [])
            )
            CONTEXT_USAGE.labels(model=model_label).observe(
                (compacted_tokens + output_budget) / effective
            )
            return compacted

    count_label = (
        "tokenizer-counted prompt" if actual_prompt_tokens is not None else "estimated prompt"
    )
    raise InferenceError(
        "CONTEXT_LIMIT_EXCEEDED",
        f"{count_label} ({prompt_tokens} tokens) plus output budget ({output_budget} tokens) "
        f"exceeds the {effective} token context window; shorten the conversation or "
        "adjust MAX_CONTEXT_LENGTH/MAX_OUTPUT_TOKENS",
        False,
        400,
    )


def estimate_prompt_tokens(messages: Any) -> int:
    """Estimate serialized prompt/tool bytes; this is not tokenizer output."""
    escaped = json.dumps(messages, ensure_ascii=True)
    prompt = json.dumps(messages, ensure_ascii=False)
    byte_estimate = (len(escaped.encode("utf-8")) + 2) // 3
    # Count every punctuation run character separately to reduce undercounting
    # code, JSON schemas and operator-heavy prompts. This remains a heuristic.
    groups = 0
    in_word = False
    for char in prompt:
        if char.isalnum() or ord(char) > 127:
            if not in_word:
                groups += 1
            in_word = True
        elif char.isspace():
            in_word = False
        else:
            groups += 1
            in_word = False
    return max(byte_estimate, groups)


async def _admit_context_window(payload: dict[str, Any], backend: Any) -> dict[str, Any]:
    model = str(payload.get("model") or settings.default_model)
    effective, output_budget = _context_limits(payload)
    estimated = estimate_prompt_tokens(_prompt_payload(payload) or payload.get("messages", []))
    if estimated + output_budget <= effective * TOKENIZER_ADMISSION_RATIO:
        # Comfortably inside the window: an exact count cannot change the
        # verdict, and the extra request would serialize on the engine's
        # tokenizer before every generation (a measurable TTFT cost for long
        # prompts). Keep the cheap heuristic for admission and metrics.
        return _check_context_window(payload, estimated_prompt_tokens=estimated)
    tokenizer = getattr(backend, "count_chat_tokens", None)
    actual = await tokenizer(model, payload) if tokenizer else None
    if tokenizer and actual is not None and actual + output_budget > effective:
        candidate = _compact_payload(payload, effective, output_budget)
        if candidate is not None:
            verified = await tokenizer(model, candidate)
            if verified is not None and verified + output_budget <= effective:
                CONTEXT_USAGE.labels(model=model).observe(
                    (verified + output_budget) / effective
                )
                return candidate
    # A heuristic must not re-admit a prompt the engine's tokenizer rejected;
    # when no exact count is available, the heuristic is still the best signal.
    return _check_context_window(
        payload,
        actual_prompt_tokens=actual,
        estimated_prompt_tokens=estimated,
        allow_compaction=actual is None,
    )


class Scheduler:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._queue = asyncio.Queue(maxsize=settings.max_queue_size)
        self._ollama = OllamaBackend()
        self._llama = LlamaCppBackend()
        self._active: dict[str, dict[str, Any]] = {}
        self._leases: dict[str, StreamLease] = {}
        self._model_active: dict[str, int] = {}
        self._request_counter = 0
        self._sweeper_task: asyncio.Task[None] | None = None
        self._runtime_maintenance = False
        self._circuits = {
            "ollama": CircuitBreaker(
                settings.circuit_failure_threshold,
                settings.circuit_recovery_seconds,
                settings.circuit_trial_timeout_seconds,
            ),
            "llama.cpp": CircuitBreaker(
                settings.circuit_failure_threshold,
                settings.circuit_recovery_seconds,
                settings.circuit_trial_timeout_seconds,
            ),
        }

    @property
    def backend(self) -> OllamaBackend | LlamaCppBackend:
        return self._llama if settings.llamacpp_base_url else self._ollama

    @property
    def circuit(self) -> CircuitBreaker:
        if self.backend.name not in self._circuits:
            self._circuits[self.backend.name] = CircuitBreaker(
                settings.circuit_failure_threshold,
                settings.circuit_recovery_seconds,
                settings.circuit_trial_timeout_seconds,
            )
        return self._circuits[self.backend.name]

    def _new_request_id(self) -> str:
        self._request_counter += 1
        return f"inf-{self._request_counter:08d}"

    def _enqueue(self) -> None:
        if self._runtime_maintenance:
            raise InferenceError(
                "RUNTIME_MAINTENANCE",
                "The inference runtime is being changed; retry shortly",
                True,
                503,
            )
        try:
            self._queue.put_nowait(object())
        except asyncio.QueueFull as exc:
            raise InferenceError("QUEUE_FULL", "Inference queue is full", True, 429) from exc
        QUEUE.set(self._queue.qsize())

    def _dequeue(self) -> None:
        """Remove one admitted request from the waiting queue without yielding."""
        self._queue.get_nowait()
        self._queue.task_done()
        QUEUE.set(self._queue.qsize())

    async def _release(self) -> None:
        """Remove a request that exits before acquiring an inference slot."""
        await self._queue.get()
        self._queue.task_done()
        QUEUE.set(self._queue.qsize())

    async def _acquire_slot(self, model: str, enqueued: float, request_id: str) -> None:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=settings.queue_timeout_seconds
            )
        except TimeoutError as exc:
            QUEUE_WAIT.labels(model=model).observe(time.perf_counter() - enqueued)
            QUEUE_TIMEOUTS.inc()
            raise InferenceError(
                "QUEUE_TIMEOUT",
                f"Inference queue wait exceeded {settings.queue_timeout_seconds:.1f}s",
                True,
                429,
            ) from exc
        QUEUE_WAIT.labels(model=model).observe(time.perf_counter() - enqueued)
        self._active[request_id] = {
            "request_id": request_id,
            "model": model,
            "backend": self.backend.name,
            "started_at": time.time(),
            "started_monotonic": time.perf_counter(),
        }
        ACTIVE_REQUESTS.inc()
        # Recompute the oldest-active gauge across ALL active entries instead
        # of stomping it to zero on every admission (GW-20).
        if self._active:
            oldest = min(entry["started_monotonic"] for entry in self._active.values())
            ACTIVE_REQUEST_AGE.set(max(0.0, time.perf_counter() - oldest))
        else:
            ACTIVE_REQUEST_AGE.set(0)

    def _release_slot(self, request_id: str, model: str) -> None:
        entry = self._active.pop(request_id, None)
        if entry is None:
            # The slot was already reclaimed by the stale sweeper (GW-15);
            # releasing again would double-count the semaphore.
            return
        self._semaphore.release()
        ACTIVE_REQUESTS.dec()
        if self._active:
            oldest = min(item["started_monotonic"] for item in self._active.values())
            ACTIVE_REQUEST_AGE.set(max(0.0, time.perf_counter() - oldest))
        else:
            ACTIVE_REQUEST_AGE.set(0)

    def _activate_model(self, model: str) -> None:
        """Count active requests per model so concurrency>1 stays accurate."""
        count = self._model_active.get(model, 0) + 1
        self._model_active[model] = count
        ACTIVE_MODEL.labels(model=model).set(count)

    def _deactivate_model(self, model: str) -> None:
        count = max(0, self._model_active.get(model, 0) - 1)
        if count:
            self._model_active[model] = count
        else:
            self._model_active.pop(model, None)
        ACTIVE_MODEL.labels(model=model).set(count)

    @property
    def stale_request_seconds(self) -> float:
        """Age after which an active slot must belong to an abandoned request.

        Chat, stream and embedding calls are all bounded by wait_for timeouts,
        so a slot held shortly beyond the longest of those timeouts can only be
        a leaked stream whose cleanup never ran. Reclaim quickly enough that a
        single abandoned stream cannot stall the only configured slot for
        minutes.
        """
        return 1.1 * max(settings.chat_timeout_seconds, settings.stream_timeout_seconds)

    def sweep_stale(self) -> list[str]:
        """Reclaim slots held by abandoned requests (GW-15).

        A streaming generator that is never consumed holds its semaphore slot
        and leaves ACTIVE_MODEL at 1 forever because its cleanup finally never
        runs. Any entry in ``_active`` older than ``stale_request_seconds``
        cannot be a legitimately running request, so it is force-released
        here. Returns the swept request ids.
        """
        limit = self.stale_request_seconds
        now = time.perf_counter()
        swept: list[str] = []
        for request_id, entry in list(self._active.items()):
            if now - entry["started_monotonic"] <= limit:
                continue
            model = entry["model"]
            self._active.pop(request_id, None)
            self._semaphore.release()
            ACTIVE_REQUESTS.dec()
            STALE_REQUESTS.inc()
            swept.append(request_id)
            lease = self._leases.pop(request_id, None)
            if lease is not None:
                # The sweep already released the slot and model gauge. Marking
                # the lease released prevents a late cleanup from decrementing
                # another live request's gauge; cancelling the driving task
                # also stops the abandoned backend generation.
                lease.released = True
                lease.cancel()
            if not any(other["model"] == model for other in self._active.values()):
                self._model_active.pop(model, None)
                ACTIVE_MODEL.labels(model=model).set(0)
        if self._active:
            oldest = min(item["started_monotonic"] for item in self._active.values())
            ACTIVE_REQUEST_AGE.set(max(0.0, time.perf_counter() - oldest))
        else:
            ACTIVE_REQUEST_AGE.set(0)
        return swept

    def start_sweeper(self) -> None:
        """Start the background loop that periodically sweeps stale slots."""
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        self._sweeper_task = asyncio.create_task(self._sweep_loop(), name="scheduler-sweeper")

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(min(30.0, max(5.0, self.stale_request_seconds / 10.0)))
            self.sweep_stale()

    def _backend_failure(self, exc: InferenceError) -> None:
        if exc.code in {
            "BACKEND_ERROR",
            "BACKEND_UNAVAILABLE",
            "MODEL_TIMEOUT",
            "EMBEDDING_TIMEOUT",
            "INFERENCE_ERROR",
        }:
            self.circuit.failure()
            BACKEND_ERRORS.labels(backend=self.backend.name, code=exc.code).inc()
        self._record_circuit()

    def _record_circuit(self) -> None:
        CIRCUIT_STATE.labels(backend=self.backend.name).set(
            1 if self.circuit.status()["state"] == "open" else 0
        )

    def _check_circuit(self) -> None:
        try:
            self.circuit.allow()
        except CircuitOpenError as exc:
            self._record_circuit()
            raise InferenceError(
                "CIRCUIT_OPEN",
                "Inference backend is temporarily unavailable",
                True,
                503,
            ) from exc

    @staticmethod
    def _record_inference_metrics(model: str, data: dict[str, Any]) -> None:
        timing = data.get("_inference") or {}
        load = float(timing.get("load_seconds") or 0)
        generation = float(timing.get("generation_seconds") or 0)
        if load:
            MODEL_LOAD.labels(model=model).observe(load)
        if generation:
            GENERATION_DURATION.labels(model=model).observe(generation)
        completion = int((data.get("usage") or {}).get("completion_tokens") or 0)
        if completion and generation:
            TOKENS_PER_SECOND.labels(model=model).observe(completion / generation)

    async def _prepare(self, model: str, *, chat: bool = True) -> bool:
        if chat:
            model_manager.validate_chat(model)
        else:
            model_manager.validate(model)
        self._check_circuit()
        return await model_manager.ensure_available(self.backend, model)

    async def embeddings(
        self, model: str, inputs: list[str], request_id: str | None = None
    ) -> list[list[float]]:
        if self.backend.name == "llama.cpp" and not settings.llamacpp_embeddings:
            raise InferenceError(
                "CAPABILITY_UNSUPPORTED",
                "The configured llama.cpp runtime does not enable embeddings",
                False,
                400,
            )
        model_manager.validate(model)
        if model != settings.embedding_model:
            raise InferenceError(
                "INVALID_EMBEDDING_MODEL",
                f"Model '{model}' is not configured as the embedding model",
                False,
                400,
            )
        if len(inputs) > settings.max_embedding_batch_size:
            raise InferenceError(
                "EMBEDDING_BATCH_TOO_LARGE",
                f"Embedding batch exceeds the limit of {settings.max_embedding_batch_size} inputs",
                False,
                400,
            )
        request_id = request_id or self._new_request_id()
        self._enqueue()
        enqueued = time.perf_counter()
        acquired = False
        queued = True
        try:
            try:
                await self._acquire_slot(model, enqueued, request_id)
                acquired = True
                self._dequeue()
                queued = False
                try:
                    warm = await self._prepare(model, chat=False)
                except InferenceError as exc:
                    # ensure_available fails before the backend call; a
                    # half-open trial started here never produced a verdict
                    # (GW-15).
                    self._backend_failure(exc)
                    raise
                try:
                    admit_request()
                except InferenceError:
                    RESOURCE_REJECTIONS.inc()
                    raise
                model_manager.activate(model, warm)
                self._activate_model(model)
                try:
                    vectors = await asyncio.wait_for(
                        self.backend.embeddings(model, inputs),
                        timeout=settings.embedding_timeout_seconds,
                    )
                    self.circuit.success()
                    self._record_circuit()
                    model_manager.record_loaded(model)
                    REQUESTS.labels(model=model, status="success").inc()
                    return vectors
                except InferenceError as exc:
                    self._backend_failure(exc)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise
                except TimeoutError as exc:
                    error = InferenceError(
                        "EMBEDDING_TIMEOUT",
                        f"Embedding request timed out after {settings.embedding_timeout_seconds:.1f}s",
                        True,
                        504,
                    )
                    self._backend_failure(error)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise error from exc
                except asyncio.CancelledError:
                    CANCELLED_REQUESTS.labels(kind="embedding").inc()
                    self.circuit.abandon_trial()
                    raise
                except Exception as exc:
                    error = InferenceError("INFERENCE_ERROR", str(exc), True, 503)
                    self._backend_failure(error)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise error from exc
                finally:
                    self._deactivate_model(model)
            except asyncio.CancelledError:
                if not acquired:
                    CANCELLED_REQUESTS.labels(kind="queued").inc()
                self.circuit.abandon_trial()
                raise
        finally:
            if acquired:
                self._release_slot(request_id, model)
            if queued:
                await self._release()

    async def chat(self, payload: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
        model = str(payload.get("model") or settings.default_model)
        model_manager.validate_chat(model)
        payload = await _admit_context_window(payload, self.backend)
        payload = dict(payload)
        payload.setdefault("max_tokens", settings.max_output_tokens)
        request_id = request_id or self._new_request_id()
        self._enqueue()
        enqueued = time.perf_counter()
        acquired = False
        queued = True
        try:
            try:
                await self._acquire_slot(model, enqueued, request_id)
                acquired = True
                self._dequeue()
                queued = False
                try:
                    warm = await self._prepare(model)
                except InferenceError as exc:
                    # ensure_available fails before the backend call; a
                    # half-open trial started here never produced a verdict
                    # (GW-15).
                    self._backend_failure(exc)
                    raise
                try:
                    admit_request()
                except InferenceError:
                    RESOURCE_REJECTIONS.inc()
                    raise
                model_manager.activate(model, warm)
                self._activate_model(model)
                try:
                    result = await asyncio.wait_for(
                        self.backend.chat(model, payload),
                        timeout=settings.chat_timeout_seconds,
                    )
                    self.circuit.success()
                    self._record_circuit()
                    model_manager.record_loaded(model)
                    usage = result.get("usage", {})
                    TOKENS.labels(model=model, kind="prompt").inc(usage.get("prompt_tokens", 0))
                    TOKENS.labels(model=model, kind="completion").inc(
                        usage.get("completion_tokens", 0)
                    )
                    self._record_inference_metrics(model, result)
                    REQUESTS.labels(model=model, status="success").inc()
                    return result
                except asyncio.CancelledError:
                    CANCELLED_REQUESTS.labels(kind="chat").inc()
                    self.circuit.abandon_trial()
                    raise
                except InferenceError as exc:
                    self._backend_failure(exc)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise
                except TimeoutError as exc:
                    error = InferenceError(
                        "MODEL_TIMEOUT",
                        f"Inference request timed out after {settings.chat_timeout_seconds:.1f}s",
                        True,
                        504,
                    )
                    self._backend_failure(error)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise error from exc
                except Exception as exc:
                    error = InferenceError("INFERENCE_ERROR", str(exc), True, 503)
                    self._backend_failure(error)
                    REQUESTS.labels(model=model, status="error").inc()
                    raise error from exc
                finally:
                    self._deactivate_model(model)
            except asyncio.CancelledError:
                if not acquired:
                    CANCELLED_REQUESTS.labels(kind="queued").inc()
                self.circuit.abandon_trial()
                raise
        finally:
            if acquired:
                self._release_slot(request_id, model)
            if queued:
                await self._release()

    async def prepare_stream(
        self, payload: dict[str, Any], request_id: str | None = None
    ) -> StreamLease:
        """Admit a stream before HTTP headers are committed by StreamingResponse."""
        model = str(payload.get("model") or settings.default_model)
        model_manager.validate_chat(model)
        payload = await _admit_context_window(payload, self.backend)
        payload = dict(payload)
        payload.setdefault("max_tokens", settings.max_output_tokens)
        request_id = request_id or self._new_request_id()
        self._enqueue()
        enqueued = time.perf_counter()
        acquired = False
        queued = True
        lease_created = False
        try:
            await self._acquire_slot(model, enqueued, request_id)
            acquired = True
            self._dequeue()
            queued = False
            try:
                warm = await self._prepare(model)
            except InferenceError as exc:
                # ensure_available fails before the backend call; a half-open
                # trial started here never produced a verdict (GW-15).
                self._backend_failure(exc)
                raise
            try:
                admit_request()
            except InferenceError:
                RESOURCE_REJECTIONS.inc()
                raise
            model_manager.activate(model, warm)
            self._activate_model(model)
            lease = StreamLease(model=model, request_id=request_id, payload=payload)
            self._leases[request_id] = lease
            lease_created = True
            return lease
        except asyncio.CancelledError:
            if not acquired:
                CANCELLED_REQUESTS.labels(kind="queued").inc()
            self.circuit.abandon_trial()
            raise
        finally:
            # Preparation failures must not leak the semaphore or queue token.
            if acquired and not lease_created:
                self._release_slot(request_id, model)
            if queued:
                await self._release()

    def release_stream(self, lease: StreamLease) -> None:
        """Release a prepared stream slot idempotently (also used by response cleanup)."""
        if lease.released:
            return
        lease.released = True
        self._leases.pop(lease.request_id, None)
        self._deactivate_model(lease.model)
        self._release_slot(lease.request_id, lease.model)

    async def stream(
        self,
        payload: dict[str, Any],
        request_id: str | None = None,
        *,
        lease: StreamLease | None = None,
    ) -> AsyncIterator[str]:
        """Return a streaming iterator, admitting the request before the first body byte."""
        prepared = lease or await self.prepare_stream(payload, request_id)
        return self._stream_prepared(prepared.payload or payload, prepared)

    async def _stream_prepared(
        self, payload: dict[str, Any], lease: StreamLease
    ) -> AsyncIterator[str]:
        model = lease.model
        payload = dict(payload)
        payload.setdefault("max_tokens", settings.max_output_tokens)
        started: float | None = None
        first_token: float | None = None
        # The stale sweeper cancels this task if the client abandons the
        # stream and normal cleanup never runs.
        lease.task = asyncio.current_task()
        try:
            started = time.perf_counter()
            saw_done = False
            try:
                async with asyncio.timeout(settings.stream_timeout_seconds):
                    async for line in self.backend.stream(model, payload):
                        now = time.perf_counter()
                        if first_token is None:
                            first_token = now
                            TTFT.labels(model=model).observe(now - started)
                        try:
                            event = json.loads(line)
                        except (TypeError, ValueError):
                            event = None
                        if not isinstance(event, dict):
                            event = None
                        if line.strip() == "[DONE]":
                            # llama.cpp ends OpenAI-style streams with the
                            # [DONE] sentinel instead of Ollama's done event
                            # (GW-21).
                            saw_done = True
                        if event and event.get("done"):
                            saw_done = True
                            load = float(event.get("load_duration") or 0) / 1e9
                            generation = float(event.get("eval_duration") or 0) / 1e9
                            completion = int(event.get("eval_count") or 0)
                            if load:
                                MODEL_LOAD.labels(model=model).observe(load)
                            if generation:
                                GENERATION_DURATION.labels(model=model).observe(generation)
                            if completion:
                                TOKENS.labels(model=model, kind="completion").inc(completion)
                                if generation:
                                    TOKENS_PER_SECOND.labels(model=model).observe(
                                        completion / generation
                                    )
                        openai_usage = event.get("usage") if event else None
                        if isinstance(openai_usage, dict) and openai_usage:
                            # OpenAI-style usage chunk (llama.cpp with
                            # stream_options.include_usage); it always
                            # immediately precedes [DONE].
                            saw_done = True
                            prompt = int(openai_usage.get("prompt_tokens") or 0)
                            completion = int(openai_usage.get("completion_tokens") or 0)
                            if prompt:
                                TOKENS.labels(model=model, kind="prompt").inc(prompt)
                            if completion:
                                TOKENS.labels(model=model, kind="completion").inc(completion)
                        # Pass the already-decoded event to the API layer so a
                        # per-token SSE line is never parsed twice.
                        yield StreamLine(line, event)
                if not saw_done:
                    # A backend stream that ends without its terminal event is
                    # an incomplete generation, not a success (GW-20).
                    raise InferenceError(
                        "STREAM_INCOMPLETE",
                        "backend stream ended without a done event",
                        True,
                        502,
                    )
                self.circuit.success()
                self._record_circuit()
                model_manager.record_loaded(model)
                REQUESTS.labels(model=model, status="success").inc()
            except asyncio.CancelledError:
                CANCELLED_REQUESTS.labels(kind="stream").inc()
                self.circuit.abandon_trial()
                raise
            except InferenceError as exc:
                self._backend_failure(exc)
                REQUESTS.labels(model=model, status="error").inc()
                raise
            except TimeoutError as exc:
                error = InferenceError(
                    "MODEL_TIMEOUT",
                    f"Streaming request timed out after {settings.stream_timeout_seconds:.1f}s",
                    True,
                    504,
                )
                self._backend_failure(error)
                REQUESTS.labels(model=model, status="error").inc()
                raise error from exc
            except Exception as exc:
                error = InferenceError("INFERENCE_ERROR", str(exc), True, 503)
                self._backend_failure(error)
                REQUESTS.labels(model=model, status="error").inc()
                raise error from exc
            finally:
                if started is not None:
                    LATENCY.labels(model=model).observe(time.perf_counter() - started)
        finally:
            self.release_stream(lease)

    def status(self) -> dict[str, Any]:
        active = []
        now = time.time()
        for item in self._active.values():
            active.append(
                {
                    "request_id": item["request_id"],
                    "model": item["model"],
                    "backend": item["backend"],
                    "age_seconds": round(max(0.0, now - item["started_at"]), 3),
                }
            )
        active.sort(key=lambda item: item["age_seconds"], reverse=True)
        return {
            "queue_depth": self._queue.qsize(),
            "queue_limit": settings.max_queue_size,
            "concurrency": settings.max_concurrent_requests,
            "available_slots": getattr(self._semaphore, "_value", None),
            "queue_timeout_seconds": settings.queue_timeout_seconds,
            "backend": self.backend.name,
            "runtime_maintenance": self._runtime_maintenance,
            "active_requests": active,
            "circuit": self.circuit.status(),
            **model_manager.status(),
        }

    def begin_runtime_maintenance(self) -> bool:
        """Block new admissions and reserve lifecycle operations when idle."""
        if self._runtime_maintenance or self._active or not self._queue.empty():
            return False
        self._runtime_maintenance = True
        return True

    def end_runtime_maintenance(self) -> None:
        self._runtime_maintenance = False

    async def readiness(self) -> dict[str, Any]:
        backend_ok = await self.backend.health()
        if not backend_ok:
            return {"ready": False, "reason": "backend_unhealthy"}
        refreshed = await model_manager.refresh(self.backend)
        if not refreshed:
            return {"ready": False, "reason": "model_state_unavailable"}
        state = model_manager.status()
        configured = set(settings.models)
        if self.backend.name == "ollama" or settings.llamacpp_embeddings:
            configured.add(settings.embedding_model)
        available = set(state["available_models"])
        missing = sorted(configured - available)
        if missing:
            return {
                "ready": False,
                "reason": "models_missing",
                "missing_models": missing,
            }
        try:
            self._check_circuit()
        except InferenceError:
            return {"ready": False, "reason": "circuit_open"}
        return {"ready": True, "missing_models": []}

    async def close(self) -> None:
        task = self._sweeper_task
        self._sweeper_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._ollama.close()
        await self._llama.close()


scheduler = Scheduler()
