import asyncio
import struct

import pytest

from inference.config import settings
from inference.llamacpp_runtime import LlamaCppRuntime, RuntimeState


def test_managed_command_carries_workstation_tuning(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "llamacpp_model", str(tmp_path / "part-00001.gguf"))
    monkeypatch.setattr(settings, "llamacpp_alias", "workstation")
    monkeypatch.setattr(settings, "max_context_length", 131072)
    monkeypatch.setattr(settings, "llamacpp_threads", 12)
    monkeypatch.setattr(settings, "llamacpp_cache_type_k", "q8_0")
    command = LlamaCppRuntime().command()
    assert command[command.index("--model") + 1].endswith("part-00001.gguf")
    assert command[command.index("--alias") + 1] == "workstation"
    assert command[command.index("--ctx-size") + 1] == "131072"
    assert command[command.index("--threads") + 1] == "12"
    assert command[command.index("--cache-type-k") + 1] == "q8_0"


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
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        async def wait(self):
            return self.returncode

    runtime = LlamaCppRuntime()
    child = Child()
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
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: {})
    monkeypatch.setattr("inference.llamacpp_runtime.kv_bytes_per_token", lambda *args: 0)
    monkeypatch.setattr("inference.llamacpp_runtime.discover_gguf", lambda path: [])
    monkeypatch.setattr("inference.llamacpp_runtime.memory_available_gb", lambda: 48.0)

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
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: {})

    with pytest.raises(RuntimeError, match="cannot estimate KV memory"):
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
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: {})
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
    monkeypatch.setattr("inference.llamacpp_runtime.read_metadata", lambda path: {})
    monkeypatch.setattr("inference.llamacpp_runtime.kv_bytes_per_token", lambda *args: 8)
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
