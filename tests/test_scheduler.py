import asyncio
import time

import pytest

from inference.config import settings
from inference.errors import InferenceError
from inference.metrics import ACTIVE_MODEL
from inference.models import model_manager
from inference.scheduler import Scheduler, _check_context_window, estimate_prompt_tokens


class FakeBackend:
    name = "fake"

    async def health(self) -> bool:
        return True

    async def available_models(self) -> list[str]:
        return ["qwen3:1.7b", "qwen3:4b", "nomic-embed-text"]

    async def loaded_models(self) -> list[str]:
        return ["qwen3:1.7b"]

    async def chat(self, model: str, payload: dict) -> dict:
        return {
            "model": model,
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in inputs]

    async def stream(self, model: str, payload: dict):
        yield '{"message":{"role":"assistant","content":"ok"},"done":true}'


class FailingBackend(FakeBackend):
    async def chat(self, model: str, payload: dict) -> dict:
        raise RuntimeError("backend exploded")


class SlowBackend(FakeBackend):
    async def chat(self, model: str, payload: dict) -> dict:
        await asyncio.sleep(0.05)
        return await super().chat(model, payload)


@pytest.fixture
def scheduler() -> Scheduler:
    instance = Scheduler()
    instance._semaphore = asyncio.Semaphore(1)
    instance._queue = asyncio.Queue(maxsize=2)
    model_manager._available.clear()
    model_manager._loaded.clear()
    model_manager._last_refresh = 0
    return instance


@pytest.mark.asyncio
async def test_chat_success_records_result(scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    result = await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert result["model"] == "qwen3:1.7b"
    assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 3}
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["warm_requests"] >= 1


@pytest.mark.asyncio
async def test_stream_uses_scheduler_lifecycle(scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    stream = await scheduler.stream({"model": "qwen3:1.7b", "messages": []})
    chunks = [chunk async for chunk in stream]

    assert chunks
    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_unknown_model_is_rejected(scheduler: Scheduler) -> None:
    with pytest.raises(InferenceError, match="Unknown model") as exc:
        await scheduler.chat({"model": "not-real", "messages": []})

    assert exc.value.code == "UNKNOWN_MODEL"


@pytest.mark.asyncio
async def test_backend_exception_is_normalized(scheduler: Scheduler) -> None:
    scheduler._ollama = FailingBackend()

    with pytest.raises(InferenceError) as exc:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert exc.value.code == "INFERENCE_ERROR"
    assert exc.value.status_code == 503
    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_timeout_is_normalized(monkeypatch, scheduler: Scheduler) -> None:
    scheduler._ollama = SlowBackend()
    monkeypatch.setattr("inference.scheduler.settings.chat_timeout_seconds", 0.001)

    with pytest.raises(InferenceError) as exc:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert exc.value.code == "MODEL_TIMEOUT"
    assert exc.value.status_code == 504
    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_queue_full_returns_retryable_error(scheduler: Scheduler) -> None:
    scheduler._queue.put_nowait(object())
    scheduler._queue.put_nowait(object())

    with pytest.raises(InferenceError) as exc:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert exc.value.code == "QUEUE_FULL"
    assert exc.value.retryable is True
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_resource_pressure_is_normalized(monkeypatch, scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    monkeypatch.setattr(
        "inference.scheduler.admit_request",
        lambda: (_ for _ in ()).throw(
            InferenceError("RESOURCE_PRESSURE", "not enough memory", True, 503)
        ),
    )

    with pytest.raises(InferenceError) as exc:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert exc.value.code == "RESOURCE_PRESSURE"
    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_readiness_reports_model_availability(scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    result = await scheduler.readiness()
    assert result["ready"] is True


@pytest.mark.asyncio
async def test_cancellation_releases_queue_and_active_slot(
    scheduler: Scheduler,
) -> None:
    class BlockingBackend(FakeBackend):
        async def chat(self, model: str, payload: dict) -> dict:
            await asyncio.sleep(10)
            return await super().chat(model, payload)

    scheduler._ollama = BlockingBackend()
    task = asyncio.create_task(scheduler.chat({"model": "qwen3:1.7b", "messages": []}))
    await asyncio.sleep(0.01)
    assert scheduler.status()["queue_depth"] == 0
    assert len(scheduler.status()["active_requests"]) == 1
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_embedding_resource_pressure_is_counted(monkeypatch, scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    monkeypatch.setattr(
        "inference.scheduler.admit_request",
        lambda: (_ for _ in ()).throw(
            InferenceError("RESOURCE_PRESSURE", "not enough memory", True, 503)
        ),
    )

    with pytest.raises(InferenceError) as exc:
        await scheduler.embeddings("nomic-embed-text", ["hello"])

    assert exc.value.code == "RESOURCE_PRESSURE"
    assert scheduler.status()["queue_depth"] == 0


@pytest.mark.asyncio
async def test_readiness_fails_when_model_state_refresh_fails(
    scheduler: Scheduler,
) -> None:
    scheduler._ollama = FakeBackend()

    async def broken_refresh(backend, force=False) -> bool:
        return False

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(model_manager, "refresh", broken_refresh)
    try:
        result = await scheduler.readiness()
    finally:
        monkeypatch.undo()

    assert result == {"ready": False, "reason": "model_state_unavailable"}


@pytest.mark.asyncio
async def test_embeddings_success_records_result(scheduler: Scheduler) -> None:
    scheduler._ollama = FakeBackend()
    result = await scheduler.embeddings("nomic-embed-text", ["hello", "world"])

    assert result == [[0.1, 0.2], [0.1, 0.2]]
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["active_requests"] == []


@pytest.mark.asyncio
async def test_embedding_timeout_is_normalized(monkeypatch, scheduler: Scheduler) -> None:
    class SlowEmbeddingBackend(FakeBackend):
        async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
            await asyncio.sleep(0.05)
            return await super().embeddings(model, inputs)

    scheduler._ollama = SlowEmbeddingBackend()
    monkeypatch.setattr("inference.scheduler.settings.embedding_timeout_seconds", 0.001)

    with pytest.raises(InferenceError) as exc:
        await scheduler.embeddings("nomic-embed-text", ["hello"])

    assert exc.value.code == "EMBEDDING_TIMEOUT"
    assert exc.value.status_code == 504
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["active_requests"] == []


@pytest.mark.asyncio
async def test_waiting_request_times_out_without_blocking(
    scheduler: Scheduler, monkeypatch
) -> None:
    class BlockingBackend(FakeBackend):
        async def chat(self, model: str, payload: dict) -> dict:
            await asyncio.sleep(1)
            return await super().chat(model, payload)

    scheduler._ollama = BlockingBackend()
    monkeypatch.setattr("inference.scheduler.settings.queue_timeout_seconds", 0.01)

    first = asyncio.create_task(scheduler.chat({"model": "qwen3:1.7b", "messages": []}))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(scheduler.embeddings("nomic-embed-text", ["hello"]))

    with pytest.raises(InferenceError) as exc:
        await second

    assert exc.value.code == "QUEUE_TIMEOUT"
    assert scheduler.status()["queue_depth"] == 0
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1


@pytest.mark.asyncio
async def test_model_preparation_waits_until_slot_is_acquired(
    scheduler: Scheduler, monkeypatch
) -> None:
    scheduler._ollama = FakeBackend()
    monkeypatch.setattr("inference.scheduler.settings.queue_timeout_seconds", 0.01)
    entered_prepare = asyncio.Event()
    allow_prepare = asyncio.Event()
    prepare_calls = 0

    async def blocked_prepare(model: str, *, chat: bool = True) -> bool:
        nonlocal prepare_calls
        prepare_calls += 1
        entered_prepare.set()
        await allow_prepare.wait()
        return True

    monkeypatch.setattr(scheduler, "_prepare", blocked_prepare)
    first = asyncio.create_task(scheduler.chat({"model": "qwen3:1.7b", "messages": []}))
    await entered_prepare.wait()
    second = asyncio.create_task(scheduler.chat({"model": "qwen3:1.7b", "messages": []}))

    with pytest.raises(InferenceError) as caught:
        await second

    assert caught.value.code == "QUEUE_TIMEOUT"
    assert prepare_calls == 1
    allow_prepare.set()
    await first
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1


@pytest.mark.asyncio
async def test_model_preparation_failure_releases_slot_and_queue(
    scheduler: Scheduler, monkeypatch
) -> None:
    async def fail_prepare(model: str, *, chat: bool = True) -> bool:
        raise InferenceError("BACKEND_UNAVAILABLE", "backend unavailable", True, 503)

    monkeypatch.setattr(scheduler, "_prepare", fail_prepare)
    with pytest.raises(InferenceError) as caught:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": []})

    assert caught.value.code == "BACKEND_UNAVAILABLE"
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1
    assert scheduler.status()["active_requests"] == []


@pytest.mark.asyncio
async def test_stream_preparation_failure_releases_slot_and_queue(
    scheduler: Scheduler, monkeypatch
) -> None:
    async def fail_prepare(model: str, *, chat: bool = True) -> bool:
        raise InferenceError("BACKEND_UNAVAILABLE", "backend unavailable", True, 503)

    monkeypatch.setattr(scheduler, "_prepare", fail_prepare)
    with pytest.raises(InferenceError) as caught:
        await scheduler.prepare_stream({"model": "qwen3:1.7b", "messages": []})

    assert caught.value.code == "BACKEND_UNAVAILABLE"
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1
    assert scheduler.status()["active_requests"] == []


@pytest.mark.asyncio
async def test_embedding_batch_limit_rejects_before_queueing(
    scheduler: Scheduler, monkeypatch
) -> None:
    monkeypatch.setattr("inference.scheduler.settings.max_embedding_batch_size", 2)

    with pytest.raises(InferenceError) as caught:
        await scheduler.embeddings("nomic-embed-text", ["one", "two", "three"])

    assert caught.value.code == "EMBEDDING_BATCH_TOO_LARGE"
    assert caught.value.status_code == 400
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1


@pytest.mark.asyncio
async def test_context_override_cannot_exceed_configured_limit(
    scheduler: Scheduler, monkeypatch
) -> None:
    monkeypatch.setattr("inference.scheduler.settings.max_context_length", 8192)

    with pytest.raises(InferenceError) as caught:
        await scheduler.chat({"model": "qwen3:1.7b", "messages": [], "num_ctx": 16384})

    assert caught.value.code == "CONTEXT_LIMIT_EXCEEDED"
    assert caught.value.status_code == 400
    assert scheduler.status()["queue_depth"] == 0
    assert scheduler.status()["available_slots"] == 1


@pytest.mark.asyncio
async def test_queue_timeout_is_included_in_wait_latency_metrics(
    scheduler: Scheduler, monkeypatch
) -> None:
    observed = []

    class HistogramChild:
        def observe(self, value: float) -> None:
            observed.append(value)

    class Histogram:
        def labels(self, **_labels):
            return HistogramChild()

    monkeypatch.setattr("inference.scheduler.QUEUE_WAIT", Histogram())
    monkeypatch.setattr("inference.scheduler.settings.queue_timeout_seconds", 0.001)
    await scheduler._semaphore.acquire()

    try:
        with pytest.raises(InferenceError) as caught:
            await scheduler._acquire_slot("qwen3:1.7b", 0.0, "queue-timeout-test")
    finally:
        scheduler._semaphore.release()

    assert caught.value.code == "QUEUE_TIMEOUT"
    assert len(observed) == 1
    assert observed[0] > 0


class IncompleteStreamBackend(FakeBackend):
    async def stream(self, model: str, payload: dict):
        yield '{"message":{"role":"assistant","content":"partial"},"done":false}'
        yield '{"message":{"role":"assistant","content":"more"},"done":false}'


@pytest.mark.asyncio
async def test_stream_without_done_event_is_a_failure(scheduler: Scheduler) -> None:
    """GW-20: a backend stream ending without `done` is an incomplete generation."""
    scheduler._ollama = IncompleteStreamBackend()

    stream = await scheduler.stream({"model": "qwen3:1.7b", "messages": []})

    chunks = []
    with pytest.raises(InferenceError) as exc:
        async for chunk in stream:
            chunks.append(chunk)
            # Nothing after the partial content; the terminal event is missing.

    assert exc.value.code == "STREAM_INCOMPLETE"
    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_active_request_age_tracks_oldest_entry(scheduler: Scheduler, monkeypatch) -> None:
    """GW-20: admitting a newer request must not stomp the oldest-age gauge."""
    from inference.metrics import ACTIVE_REQUEST_AGE
    from inference.scheduler import InferenceError  # noqa: F401

    scheduler._semaphore = asyncio.Semaphore(2)
    scheduler._queue = asyncio.Queue(maxsize=4)
    model_manager._available.add("qwen3:1.7b")
    model_manager._loaded.add("qwen3:1.7b")

    clock = {"value": 0.0}

    def fake_perf_counter() -> float:
        return clock["value"]

    monkeypatch.setattr("inference.scheduler.time.perf_counter", fake_perf_counter)
    clock["value"] = 9.0
    await scheduler._acquire_slot("qwen3:1.7b", 8.5, "req-old")
    # At admission the request itself is brand new, so the gauge is ~0.

    clock["value"] = 11.0
    await scheduler._acquire_slot("qwen3:1.7b", 10.5, "req-new")
    reading = ACTIVE_REQUEST_AGE._value.get()
    # Two seconds elapsed since req-old started: the gauge must still be
    # driven by the OLD request, not stomped to 0 by the new admission.
    assert abs(reading - 2.0) < 1e-6


def _active_entry(request_id: str, age_seconds: float) -> dict:
    """Build an _active entry that looks admitted `age_seconds` ago."""
    return {
        "request_id": request_id,
        "model": "qwen3:1.7b",
        "backend": "fake",
        "started_at": time.time() - age_seconds,
        "started_monotonic": time.perf_counter() - age_seconds,
    }


@pytest.mark.asyncio
async def test_sweep_stale_reclaims_abandoned_slot(scheduler: Scheduler) -> None:
    """GW-15: a slot held longer than 1.5x the max timeout belongs to an abandoned stream."""
    stale_age = 1.5 * max(settings.chat_timeout_seconds, settings.stream_timeout_seconds) + 10
    await scheduler._semaphore.acquire()
    scheduler._active["inf-1"] = _active_entry("inf-1", stale_age)
    ACTIVE_MODEL.labels(model="qwen3:1.7b").set(1)

    swept = scheduler.sweep_stale()

    assert swept == ["inf-1"]
    assert scheduler._active == {}
    assert scheduler._semaphore._value == 1
    assert ACTIVE_MODEL.labels(model="qwen3:1.7b")._value.get() == 0


@pytest.mark.asyncio
async def test_sweep_stale_leaves_running_requests(scheduler: Scheduler) -> None:
    """GW-15: entries younger than the stale threshold are untouched."""
    await scheduler._semaphore.acquire()
    scheduler._active["inf-1"] = _active_entry("inf-1", 0.0)

    swept = scheduler.sweep_stale()

    assert swept == []
    assert "inf-1" in scheduler._active
    scheduler._release_slot("inf-1", "qwen3:1.7b")


@pytest.mark.asyncio
async def test_release_slot_after_sweep_does_not_over_release(
    scheduler: Scheduler,
) -> None:
    """GW-15: the abandoned handler's late finally must not double-release."""
    stale_age = 1.5 * max(settings.chat_timeout_seconds, settings.stream_timeout_seconds) + 10
    await scheduler._semaphore.acquire()
    scheduler._active["inf-1"] = _active_entry("inf-1", stale_age)

    scheduler.sweep_stale()
    assert scheduler._semaphore._value == 1
    scheduler._release_slot("inf-1", "qwen3:1.7b")
    assert scheduler._semaphore._value == 1


@pytest.mark.asyncio
async def test_sweep_keeps_active_model_gauge_for_live_requests(
    scheduler: Scheduler,
) -> None:
    """GW-15: sweeping one stale entry must not stomp a live request's model gauge."""
    scheduler._semaphore = asyncio.Semaphore(2)
    stale_age = 1.5 * max(settings.chat_timeout_seconds, settings.stream_timeout_seconds) + 10
    await scheduler._semaphore.acquire()
    await scheduler._semaphore.acquire()
    scheduler._active["inf-1"] = _active_entry("inf-1", stale_age)
    scheduler._active["inf-2"] = _active_entry("inf-2", 0.0)
    ACTIVE_MODEL.labels(model="qwen3:1.7b").set(1)

    swept = scheduler.sweep_stale()

    assert swept == ["inf-1"]
    assert ACTIVE_MODEL.labels(model="qwen3:1.7b")._value.get() == 1
    scheduler._release_slot("inf-2", "qwen3:1.7b")


@pytest.mark.asyncio
async def test_sweeper_task_starts_and_close_cancels_it(scheduler: Scheduler) -> None:
    """GW-15: the background sweeper is idempotent to start and cancelled on close."""
    scheduler.start_sweeper()
    task = scheduler._sweeper_task
    assert task is not None and not task.done()
    scheduler.start_sweeper()  # must not spawn a second loop
    assert scheduler._sweeper_task is task

    await scheduler.close()

    assert task.done()
    assert scheduler._sweeper_task is None


class OpenAIStyleBackend(FakeBackend):
    """A llama.cpp-style backend that streams OpenAI-format SSE chunks."""

    name = "llama.cpp"

    async def stream(self, model: str, payload: dict):
        yield '{"choices":[{"delta":{"content":"hi"}}]}'
        yield (
            '{"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        )
        yield "[DONE]"


@pytest.mark.asyncio
async def test_openai_style_stream_completes_without_stream_incomplete(
    scheduler: Scheduler, monkeypatch
) -> None:
    """GW-21: llama.cpp streams end with [DONE], not Ollama's done event.

    Before the fix, `saw_done` stayed False and every llama.cpp stream was
    reported as STREAM_INCOMPLETE (502) after generating normally.
    """
    monkeypatch.setattr(settings, "llamacpp_url", "http://llama")
    scheduler._llama = OpenAIStyleBackend()
    lease = await scheduler.prepare_stream({"model": "qwen3:1.7b", "messages": []})

    lines = []
    async for line in scheduler._stream_prepared({"model": "qwen3:1.7b"}, lease):
        lines.append(line)

    assert lines[-1] == "[DONE]"
    assert scheduler.status()["queue_depth"] == 0


def test_context_admission_supports_128k_and_reserves_output(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_context_length", 131072)
    monkeypatch.setattr(settings, "max_output_tokens", 8192)

    _check_context_window(
        {
            "model": "qwen3:1.7b",
            "messages": [{"role": "user", "content": "x" * 90000}],
            "num_ctx": 131072,
            "max_tokens": 8192,
        }
    )
    with pytest.raises(InferenceError, match=r"prompt \("):
        _check_context_window(
            {
                "model": "qwen3:1.7b",
                "messages": [{"role": "user", "content": "x" * 400000}],
                "num_ctx": 131072,
                "max_tokens": 8192,
            }
        )


class TokenCountingBackend(FakeBackend):
    def __init__(self) -> None:
        self.counts = 0

    async def count_chat_tokens(self, model: str, payload: dict) -> int:
        self.counts += 1
        return 1


@pytest.mark.asyncio
async def test_context_admission_skips_tokenizer_far_below_window(
    monkeypatch, scheduler: Scheduler
) -> None:
    """The extra engine tokenization round-trip only runs near the limit."""
    monkeypatch.setattr(settings, "max_context_length", 8192)
    monkeypatch.setattr(settings, "max_output_tokens", 512)
    backend = TokenCountingBackend()
    scheduler._ollama = backend

    await scheduler.chat(
        {"model": "qwen3:1.7b", "messages": [{"role": "user", "content": "hello"}]}
    )

    assert backend.counts == 0


@pytest.mark.asyncio
async def test_context_admission_uses_tokenizer_near_window(
    monkeypatch, scheduler: Scheduler
) -> None:
    monkeypatch.setattr(settings, "max_context_length", 1024)
    monkeypatch.setattr(settings, "max_output_tokens", 64)
    backend = TokenCountingBackend()
    scheduler._ollama = backend

    await scheduler.chat(
        {"model": "qwen3:1.7b", "messages": [{"role": "user", "content": "x" * 4000}]}
    )

    assert backend.counts == 1


@pytest.mark.asyncio
async def test_active_model_gauge_tracks_concurrency(monkeypatch, scheduler: Scheduler) -> None:
    class BarrierBackend(FakeBackend):
        def __init__(self) -> None:
            self.seen: list[int] = []
            self._arrived = 0
            self._both_running = asyncio.Event()

        async def chat(self, model: str, payload: dict) -> dict:
            self.seen.append(ACTIVE_MODEL.labels(model=model)._value.get())
            self._arrived += 1
            if self._arrived >= 2:
                self._both_running.set()
            await self._both_running.wait()
            return await super().chat(model, payload)

    backend = BarrierBackend()
    scheduler._ollama = backend
    scheduler._semaphore = asyncio.Semaphore(2)

    await asyncio.gather(
        scheduler.chat({"model": "qwen3:1.7b", "messages": []}),
        scheduler.chat({"model": "qwen3:1.7b", "messages": []}),
    )

    assert max(backend.seen) == 2
    assert ACTIVE_MODEL.labels(model="qwen3:1.7b")._value.get() == 0


def test_context_heuristic_handles_multilingual_and_code_as_estimates(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_context_length", 8192)
    monkeypatch.setattr(settings, "max_output_tokens", 512)
    multilingual = [{"role": "user", "content": "こんにちは 世界 مرحبا بالعالم"}]
    code = [
        {
            "role": "user",
            "content": "def f(x):\n    return {'a': x**2, 'b': [i for i in range(20)]}",
        }
    ]

    assert estimate_prompt_tokens(multilingual) > 0
    assert estimate_prompt_tokens(code) > 0
    assert estimate_prompt_tokens(
        {"messages": [], "tools": [{"name": "search", "schema": "x" * 90}]}
    ) > estimate_prompt_tokens({"messages": []})
    # A tokenizer result, when provided, is used as an actual count and named
    # separately from the byte-length fallback in admission diagnostics.
    with pytest.raises(InferenceError, match="tokenizer-counted prompt"):
        _check_context_window(
            {"messages": code, "num_ctx": 100, "max_tokens": 20},
            actual_prompt_tokens=90,
        )


def test_runtime_maintenance_blocks_inference_admission() -> None:
    scheduler = Scheduler()
    assert scheduler.begin_runtime_maintenance()
    with pytest.raises(InferenceError) as caught:
        scheduler._enqueue()
    assert caught.value.code == "RUNTIME_MAINTENANCE"
    scheduler.end_runtime_maintenance()
    scheduler._enqueue()
    assert scheduler.status()["queue_depth"] == 1


@pytest.mark.asyncio
async def test_llamacpp_embeddings_require_explicit_capability(monkeypatch) -> None:
    scheduler = Scheduler()
    monkeypatch.setattr(settings, "llamacpp_url", "http://llama")
    monkeypatch.setattr(settings, "llamacpp_managed", False)
    monkeypatch.setattr(settings, "llamacpp_embeddings", False)

    with pytest.raises(InferenceError) as caught:
        await scheduler.embeddings(settings.embedding_model, ["hello"])

    assert caught.value.code == "CAPABILITY_UNSUPPORTED"
    await scheduler.close()


@pytest.mark.asyncio
async def test_oversized_chat_forwards_compacted_payload(scheduler: Scheduler, monkeypatch) -> None:
    """Compaction must change the payload the backend actually receives."""
    monkeypatch.setattr(settings, "max_context_length", 512)
    monkeypatch.setattr(settings, "max_output_tokens", 16)

    class CapturingBackend(FakeBackend):
        def __init__(self) -> None:
            self.payload: dict | None = None

        async def chat(self, model: str, payload: dict) -> dict:
            self.payload = dict(payload)
            return await super().chat(model, payload)

    backend = CapturingBackend()
    scheduler._ollama = backend
    await scheduler.chat(
        {
            "model": "qwen3:1.7b",
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "x" * 3000},
                {"role": "user", "content": "NEW"},
            ],
        }
    )

    assert backend.payload is not None
    assert [message["content"] for message in backend.payload["messages"]] == ["SYS", "NEW"]
