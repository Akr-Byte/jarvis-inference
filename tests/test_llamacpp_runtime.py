import asyncio
import struct

import pytest

from inference.config import settings
from inference.llamacpp_runtime import LlamaCppRuntime, RuntimeState


def _metadata(context: int = 32768) -> dict[str, object]:
    return {
        "general.architecture": "llama",
        "llama.context_length": context,
        "llama.block_count": 2,
        "llama.embedding_length": 64,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "llama.attention.key_length": 16,
        "llama.attention.value_length": 16,
    }


def test_managed_command_carries_workstation_tuning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "part-00001.gguf"))
    monkeypatch.setattr(settings, "llamacpp_alias", "workstation")
    monkeypatch.setattr(settings, "max_context_length", 131072)
    monkeypatch.setattr(settings, "llamacpp_parallel", 2)
    monkeypatch.setattr(settings, "llamacpp_threads", 12)
    monkeypatch.setattr(settings, "llamacpp_cache_type_k", "q8_0")
    command = LlamaCppRuntime().command()
    assert command[command.index("--model") + 1].endswith("part-00001.gguf")
    assert command[command.index("--alias") + 1] == "workstation"
    assert command[command.index("--ctx-size") + 1] == "262144"
    assert command[command.index("--threads") + 1] == "12"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"


def test_managed_command_passes_engine_parity_flags(monkeypatch, tmp_path) -> None:
    """Chat templates, metrics, mlock and reasoning separation are opt-in flags."""
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_alias", "parity")
    monkeypatch.setattr(settings, "llamacpp_jinja", True)
    monkeypatch.setattr(settings, "llamacpp_metrics", True)
    monkeypatch.setattr(settings, "llamacpp_mlock", True)
    monkeypatch.setattr(settings, "llamacpp_reasoning_format", "auto")
    monkeypatch.setattr(settings, "llamacpp_cache_reuse", 256)
    monkeypatch.setattr(settings, "llamacpp_gpu_layers", 99)
    runtime.supported_flags = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--batch-size",
        "--ubatch-size",
        "--jinja",
        "--metrics",
        "--mlock",
        "--reasoning-format",
        "--cache-reuse",
        "--n-gpu-layers",
    }

    command = runtime.command()

    assert "--jinja" in command
    assert "--metrics" in command
    assert "--mlock" in command
    assert command[command.index("--reasoning-format") + 1] == "auto"
    assert command[command.index("--cache-reuse") + 1] == "256"
    assert command[command.index("--n-gpu-layers") + 1] == "99"


def test_managed_command_skips_parity_flags_unsupported_by_engine(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_jinja", True)
    monkeypatch.setattr(settings, "llamacpp_metrics", True)
    monkeypatch.setattr(settings, "llamacpp_reasoning_format", "auto")
    runtime.supported_flags = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--batch-size",
        "--ubatch-size",
    }

    command = runtime.command()

    assert "--jinja" not in command
    assert "--metrics" not in command
    assert "--reasoning-format" not in command


@pytest.mark.asyncio
async def test_unmanaged_runtime_does_not_spawn_or_signal_process(monkeypatch) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_managed", False)

    async def unexpected(*args, **kwargs):
        raise AssertionError("unmanaged mode must not spawn a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)
    await runtime.start()
    await runtime.stop()
    assert runtime.process is None


@pytest.mark.asyncio
async def test_stop_terminates_only_owned_child(monkeypatch, tmp_path) -> None:
    class Child:
        pid = 4321
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        async def wait(self):
            return self.returncode

    runtime = LlamaCppRuntime()
    child = Child()

    def killpg(pid, sig):
        if sig == 15:
            child.terminate()
        elif sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr("inference.llamacpp_runtime.os.killpg", killpg)
    runtime.process = child
    runtime._log = (tmp_path / "server.log").open("wb")
    await runtime.stop()
    assert child.terminated
    assert runtime.process is None


@pytest.mark.asyncio
async def test_start_creates_managed_child_and_reports_owned_pid(monkeypatch, tmp_path) -> None:
    class Child:
        pid = 4321
        returncode = None

        def __init__(self):
            self.exited = asyncio.Event()
            self.stdout = asyncio.StreamReader()

        def terminate(self):
            self.returncode = 0
            self.exited.set()
            self.stdout.feed_eof()

        async def wait(self):
            await self.exited.wait()
            return self.returncode

    child = Child()
    runtime = LlamaCppRuntime()
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_model", str(model))
    monkeypatch.setattr(settings, "llamacpp_alias", "tiny")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setattr(settings, "llamacpp_log_file", str(tmp_path / "llama.log"))
    monkeypatch.setattr(runtime, "detect_capabilities", _async_value({}))
    monkeypatch.setattr(runtime, "_wait_healthy", _async_value(None))
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: _metadata())
    monkeypatch.setattr("inference.llamacpp_runtime.discover_gguf", lambda path: [])
    monkeypatch.setattr("inference.llamacpp_runtime.memory_available_gb", lambda: 48.0)
    monkeypatch.setattr(
        "inference.llamacpp_runtime.os.killpg",
        lambda pid, sig: (
            child.terminate() if sig == 15 else (_ for _ in ()).throw(ProcessLookupError())
        ),
    )

    async def spawn(*args, **kwargs):
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    await runtime.start()

    assert runtime.status()["pid"] == 4321
    assert runtime.status()["owns_process"] is True
    assert runtime.state.value == "ready"
    await runtime.stop()
    assert child.returncode == 0


@pytest.mark.asyncio
async def test_shutdown_during_startup_waits_for_owned_child_and_stops_it(
    monkeypatch, tmp_path
) -> None:
    class Child:
        pid = 9876
        returncode = None

        def __init__(self):
            self.exited = asyncio.Event()
            self.stdout = asyncio.StreamReader()

        def terminate(self):
            self.returncode = 0
            self.exited.set()
            self.stdout.feed_eof()

        async def wait(self):
            await self.exited.wait()
            return self.returncode

    child = Child()
    entered_health = asyncio.Event()
    release_health = asyncio.Event()
    runtime = LlamaCppRuntime()
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_model", str(model))
    monkeypatch.setattr(settings, "llamacpp_alias", "tiny")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setattr(settings, "llamacpp_log_file", str(tmp_path / "llama.log"))
    monkeypatch.setattr(runtime, "detect_capabilities", _async_value({}))
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: _metadata())
    monkeypatch.setattr("inference.llamacpp_runtime.discover_gguf", lambda path: [])
    monkeypatch.setattr("inference.llamacpp_runtime.memory_available_gb", lambda: 48.0)

    async def wait_for_health():
        entered_health.set()
        await release_health.wait()

    async def spawn(*args, **kwargs):
        return child

    def killpg(pid, sig):
        if sig == 15:
            child.terminate()
        elif sig == 0:
            raise ProcessLookupError

    monkeypatch.setattr(runtime, "_wait_healthy", wait_for_health)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("inference.llamacpp_runtime.os.killpg", killpg)
    start_task = asyncio.create_task(runtime.start())
    await entered_health.wait()
    stop_task = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    release_health.set()
    await asyncio.gather(start_task, stop_task)

    assert runtime.state is RuntimeState.STOPPED
    assert runtime.process is None
    assert child.returncode == 0


@pytest.mark.asyncio
async def test_runtime_restart_limit_leaves_failed_state(monkeypatch) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_restart_enabled", False)

    class ExitedChild:
        returncode = 9

        async def wait(self):
            return 9

    runtime.process = ExitedChild()
    runtime.state = RuntimeState.READY
    runtime._stopping = False
    await runtime._watch()

    assert runtime.state.value == "failed"
    assert runtime.status()["restart_count"] == 1


@pytest.mark.asyncio
async def test_start_rejects_context_when_gguf_memory_metadata_is_missing(
    monkeypatch, tmp_path
) -> None:
    runtime = LlamaCppRuntime()
    model = tmp_path / "unknown-model.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_model", str(model))
    monkeypatch.setattr(settings, "llamacpp_alias", "unknown-model")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setattr(settings, "max_context_length", 131072)
    monkeypatch.setattr(runtime, "detect_capabilities", _async_value({}))
    monkeypatch.setattr(
        "inference.llamacpp_runtime.read_metadata",
        lambda path: {"general.architecture": "llama", "llama.context_length": 262144},
    )

    with pytest.raises(RuntimeError, match="embedding_length"):
        await runtime.start()

    assert runtime.state.value == "failed"
    assert runtime.process is None


@pytest.mark.asyncio
async def test_start_rejects_incomplete_split_model(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    model = tmp_path / "model-00001-of-00002.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_model", str(model))
    monkeypatch.setattr(settings, "llamacpp_alias", "model")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setattr(runtime, "detect_capabilities", _async_value({}))
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: _metadata())
    monkeypatch.setattr("inference.llamacpp_runtime.discover_gguf", lambda path: [])

    with pytest.raises(RuntimeError, match="split GGUF shards are incomplete"):
        await runtime.start()

    assert runtime.process is None


@pytest.mark.asyncio
async def test_start_requires_trained_context_metadata_above_32k(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    model = tmp_path / "unknown-context.gguf"
    model.write_bytes(b"gguf")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_model", str(model))
    monkeypatch.setattr(settings, "llamacpp_alias", "unknown-context")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setattr(settings, "max_context_length", 131072)
    monkeypatch.setattr(runtime, "detect_capabilities", _async_value({}))
    monkeypatch.setattr(
        "inference.llamacpp_runtime.read_metadata", lambda path: _metadata(context=0)
    )
    monkeypatch.setattr("inference.llamacpp_runtime.discover_gguf", lambda path: [])

    with pytest.raises(RuntimeError, match="does not declare its trained context"):
        await runtime.start()


def _async_value(value):
    async def return_value(*args, **kwargs):
        return value

    return return_value


def _gguf_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def test_gguf_metadata_reads_model_context(tmp_path) -> None:
    from inference.gguf import read_metadata

    path = tmp_path / "tiny.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 2)
        + _gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + _gguf_string("qwen3")
        + _gguf_string("qwen3.context_length")
        + struct.pack("<II", 4, 131072)
    )
    assert read_metadata(path)["qwen3.context_length"] == 131072


def _gguf_string_array(values: list[str]) -> bytes:
    return struct.pack("<IQ", 8, len(values)) + b"".join(_gguf_string(value) for value in values)


def test_gguf_metadata_skips_unwanted_string_arrays(tmp_path) -> None:
    """Tokenizer vocab arrays must be skipped, not decoded, during header scans."""
    from inference.gguf import read_metadata

    path = tmp_path / "vocab.gguf"
    path.write_bytes(
        b"GGUF"
        + struct.pack("<IQQ", 3, 0, 3)
        + _gguf_string("tokenizer.ggml.tokens")
        + struct.pack("<I", 9)
        + _gguf_string_array([f"token-{index}" for index in range(2048)])
        + _gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + _gguf_string("llama")
        + _gguf_string("llama.context_length")
        + struct.pack("<II", 4, 8192)
    )

    metadata = read_metadata(path)

    assert metadata["general.architecture"] == "llama"
    assert metadata["llama.context_length"] == 8192
    assert "tokenizer.ggml.tokens" not in metadata


def test_gguf_metadata_cache_avoids_reparsing_and_invalidates(tmp_path, monkeypatch) -> None:
    from inference import gguf

    path = tmp_path / "cached.gguf"

    def write(context: int, extra: str | None = None) -> None:
        entries = (_gguf_string("llama.context_length") + struct.pack("<II", 4, context)) + (
            _gguf_string(extra) + struct.pack("<I", 8) + _gguf_string("x") if extra else b""
        )
        count = 2 if extra else 1
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, count) + entries)

    calls = 0
    original = gguf._read_metadata_file

    def counting(source):
        nonlocal calls
        calls += 1
        return original(source)

    gguf._read_metadata_cached.cache_clear()
    monkeypatch.setattr(gguf, "_read_metadata_file", counting)

    write(4096)
    assert gguf.read_metadata(path)["llama.context_length"] == 4096
    assert gguf.read_metadata(path)["llama.context_length"] == 4096
    assert calls == 1

    write(8192, extra="general.name")
    assert gguf.read_metadata(path)["llama.context_length"] == 8192
    assert calls == 2
    gguf._read_metadata_cached.cache_clear()


def test_qwen3next_hybrid_kv_and_recurrent_state_estimate() -> None:
    from inference.gguf import estimate_kv_memory, kv_bytes_per_token

    recurrent = [index % 4 != 3 for index in range(48)]
    metadata = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 48,
        "qwen3next.embedding_length": 2048,
        "qwen3next.attention.head_count": 16,
        "qwen3next.attention.head_count_kv": 2,
        "qwen3next.attention.key_length": 256,
        "qwen3next.attention.value_length": 256,
        "qwen3next.attention.recurrent_layers": recurrent,
        "qwen3next.ssm.conv_kernel": 4,
        "qwen3next.ssm.inner_size": 4096,
        "qwen3next.ssm.state_size": 128,
        "qwen3next.ssm.time_step_rank": 32,
        "qwen3next.ssm.group_count": 16,
    }

    assert kv_bytes_per_token(metadata) == 12 * 2 * (256 * 2 + 256 * 2)
    estimated = estimate_kv_memory(metadata, context_tokens=131072)
    assert estimated.complete
    assert estimated.attention_layers == 12
    assert estimated.recurrent_layers == 36
    assert estimated.recurrent_state_bytes_per_sequence == 36 * 4 * (
        4 * (2 * 128 * 16 + 2 * 4096) + 128 * 4096
    )
    assert estimated.bytes_total == 24576 * 131072 + estimated.recurrent_state_bytes_per_sequence

    interval_metadata = dict(metadata)
    del interval_metadata["qwen3next.attention.recurrent_layers"]
    interval_metadata["qwen3next.full_attention_interval"] = 4
    interval_estimate = estimate_kv_memory(interval_metadata, context_tokens=131072)
    assert interval_estimate == estimated

    quantized = estimate_kv_memory(
        metadata, context_tokens=131072, cache_type="q8_0", cache_type_v="q8_0"
    )
    assert quantized.complete
    assert quantized.bytes_total < estimated.bytes_total


def test_qwen3next_requires_hybrid_layout_and_ssm_metadata() -> None:
    from inference.gguf import estimate_kv_memory

    metadata = {
        "general.architecture": "qwen3next",
        "qwen3next.block_count": 48,
        "qwen3next.attention.head_count": 16,
        "qwen3next.attention.head_count_kv": 2,
        "qwen3next.attention.key_length": 256,
        "qwen3next.attention.value_length": 256,
    }
    estimate = estimate_kv_memory(metadata, context_tokens=32768)
    assert not estimate.complete
    assert estimate.bytes_total is None


def test_other_hybrid_architectures_do_not_fall_back_to_transformer_kv() -> None:
    from inference.gguf import kv_bytes_per_token

    assert (
        kv_bytes_per_token(
            {
                "general.architecture": "mamba2",
                "mamba2.block_count": 48,
                "mamba2.attention.head_count_kv": 2,
                "mamba2.attention.key_length": 128,
                "mamba2.attention.value_length": 128,
            }
        )
        is None
    )


def test_memory_estimate_lists_components_and_rejects_invalid_batch(monkeypatch) -> None:
    from inference.llamacpp_runtime import estimate_runtime_memory, require_memory_capacity

    monkeypatch.setattr("inference.llamacpp_runtime.memory_available_gb", lambda: 64.0)
    monkeypatch.setattr(settings, "memory_budget_gb", 1.0)
    estimate = estimate_runtime_memory(
        model_bytes=1024**3,
        metadata=_metadata(),
        context_tokens=4096,
        parallel_sequences=1,
        batch_size=512,
        ubatch_size=128,
        cache_ram_bytes=256 * 1024**2,
        prompt_cache_enabled=True,
    )
    assert estimate.components_bytes["model_weights"] == 1024**3
    assert estimate.components_bytes["model_load_alignment_reserve"] == int(0.1 * 1024**3)
    assert estimate.components_bytes["kv_and_recurrent_state"] > 0
    assert estimate.components_bytes["prompt_cache_reserve"] == 256 * 1024**2
    assert estimate.components_bytes["batch_buffers"] > 0
    assert estimate.components_bytes["compute_buffers"] > 0
    assert estimate.components_bytes["runtime_overhead"] > 0
    assert estimate.estimated_bytes == sum(estimate.components_bytes.values())
    assert not estimate.complete
    assert estimate.headroom_bytes > 0
    with pytest.raises(RuntimeError, match="exceeds admission capacity"):
        require_memory_capacity(estimate)

    with pytest.raises(ValueError, match="micro-batch"):
        estimate_runtime_memory(
            model_bytes=1,
            metadata=_metadata(),
            context_tokens=4096,
            parallel_sequences=1,
            batch_size=64,
            ubatch_size=128,
            cache_ram_bytes=0,
            prompt_cache_enabled=False,
        )


def test_discovery_groups_complete_split_gguf_shards(tmp_path) -> None:
    from inference.model_registry import discover_gguf

    for index in (1, 2):
        (tmp_path / f"coder-IQ3_M-{index:05d}-of-00002.gguf").write_bytes(b"not-a-gguf")
    found = discover_gguf(tmp_path)
    assert len(found) == 1
    assert found[0].id == "coder-IQ3_M"
    assert len(found[0].shards) == 2
    assert found[0].size_bytes == len(b"not-a-gguf") * 2


def test_runtime_command_skips_unsupported_optional_flags(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_cache_prompt", True)
    monkeypatch.setattr(settings, "llamacpp_cache_ram_mib", 4096)
    runtime.supported_flags = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--batch-size",
        "--ubatch-size",
    }

    command = runtime.command()

    assert "--cache-prompt" not in command
    assert "--cache-ram" not in command


def test_managed_command_passes_speculative_decoding_flag(monkeypatch, tmp_path) -> None:
    """LLAMACPP_DRAFT_MODEL maps to llama-server --draft-model when supported."""
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_draft_model", "/models/draft-Qwen3-1.5B.gguf")
    monkeypatch.setattr(settings, "llamacpp_draft_max_len", 7)
    monkeypatch.setattr(settings, "llamacpp_draft_min_len", 3)
    runtime.supported_flags = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--batch-size",
        "--ubatch-size",
        "--draft-model",
        "--draft-max-len",
        "--draft-min-len",
    }

    command = runtime.command()

    assert "--draft-model" in command
    assert command[command.index("--draft-model") + 1] == "/models/draft-Qwen3-1.5B.gguf"
    assert command[command.index("--draft-max-len") + 1] == "7"
    assert command[command.index("--draft-min-len") + 1] == "3"


def test_managed_command_skips_draft_flag_when_unsupported(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_draft_model", "/models/draft-Qwen3-1.5B.gguf")
    monkeypatch.setattr(settings, "llamacpp_draft_max_len", 7)
    monkeypatch.setattr(settings, "llamacpp_draft_min_len", 3)
    runtime.supported_flags = {
        "--model",
        "--host",
        "--port",
        "--ctx-size",
        "--threads",
        "--batch-size",
        "--ubatch-size",
    }

    command = runtime.command()

    assert "--draft-model" not in command
    assert "--draft-max-len" not in command
    assert "--draft-min-len" not in command


def test_managed_command_omits_draft_flags_when_unset(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "model.gguf"))
    monkeypatch.setattr(settings, "llamacpp_draft_model", "")
    monkeypatch.setattr(settings, "llamacpp_draft_max_len", 7)
    monkeypatch.setattr(settings, "llamacpp_draft_min_len", 3)

    command = runtime.command()

    assert "--draft-model" not in command
    assert "--draft-max-len" not in command
    assert "--draft-min-len" not in command


def test_llamacpp_capabilities_expose_draft_model(monkeypatch) -> None:
    runtime = LlamaCppRuntime()
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_metrics", True)
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")

    runtime.supported_flags = {"--model", "--host", "--port", "--ctx-size", "--threads"}

    capabilities = runtime.capabilities()
    assert capabilities["draft_model"] is False
    assert capabilities["draft_max_len"] is False
    assert capabilities["draft_min_len"] is False
    assert capabilities["engine_metrics_url"].endswith("/metrics")


@pytest.mark.asyncio
async def test_log_capture_rotates_while_runtime_is_running(monkeypatch, tmp_path) -> None:
    runtime = LlamaCppRuntime()
    path = tmp_path / "runtime.log"
    runtime._log_path = path
    runtime._log = path.open("wb", buffering=0)
    runtime._log_size = 0
    monkeypatch.setattr(settings, "llamacpp_log_max_mib", 1)
    stream = asyncio.StreamReader()
    stream.feed_data(b"x" * (2 * 1024 * 1024))
    stream.feed_eof()

    await runtime._capture_logs(stream)
    runtime._log.close()

    assert path.with_suffix(".log.1").is_file()
    assert path.stat().st_size <= 1024 * 1024
