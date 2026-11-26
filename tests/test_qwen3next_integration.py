"""Opt-in smoke/performance validation for a locally available Qwen3-Next GGUF.

Run with LLAMACPP_INTEGRATION=1, LLAMACPP_INTEGRATION_MODEL=/path/model.gguf,
and LLAMACPP_INTEGRATION_SERVER=/path/llama-server. The test never downloads models.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import time
from pathlib import Path

import httpx
import psutil
import pytest

from inference.gguf import read_metadata
from inference.llamacpp_runtime import estimate_runtime_memory


@pytest.mark.asyncio
@pytest.mark.integration
async def test_local_qwen3next_contexts_and_performance(tmp_path, monkeypatch) -> None:
    if os.getenv("LLAMACPP_INTEGRATION", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("set LLAMACPP_INTEGRATION=1 to run the local-model integration test")
    model_value = os.getenv("LLAMACPP_INTEGRATION_MODEL") or os.getenv("LLAMACPP_MODEL")
    model = Path(model_value or "").expanduser()
    if not model.is_file():
        pytest.skip("configured local Qwen3-Next GGUF is unavailable; no model is downloaded")
    server_value = os.getenv("LLAMACPP_INTEGRATION_SERVER", "llama-server")
    server = shutil.which(server_value) if "/" not in server_value else server_value
    if not server or not Path(server).is_file():
        pytest.skip("llama-server is unavailable; no performance validation was run")

    metadata = read_metadata(model)
    if metadata.get("general.architecture") != "qwen3next":
        pytest.skip("configured GGUF architecture is not qwen3next")
    max_context = int(metadata.get("qwen3next.context_length", 0))
    contexts = [value for value in (16384, 32768, 65536, 131072) if value <= max_context]
    requested = os.getenv("LLAMACPP_INTEGRATION_CONTEXTS")
    if requested:
        contexts = [int(value) for value in requested.split(",") if int(value) in contexts]
    if not contexts:
        pytest.skip("model metadata does not allow any configured validation context")

    options = await asyncio.create_subprocess_exec(
        server, "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    version_out, _ = await options.communicate()
    build = version_out.decode("utf-8", errors="replace").strip()
    digest = await asyncio.to_thread(_sha256, model)
    results = []
    headroom_gb = float(os.getenv("LLAMACPP_INTEGRATION_HEADROOM_GB", "2"))
    memory_available = psutil.virtual_memory().available / 1024**3
    monkeypatch.setattr("inference.llamacpp_runtime.settings.memory_budget_gb", memory_available)
    monkeypatch.setattr("inference.llamacpp_runtime.settings.memory_headroom_gb", headroom_gb)
    monkeypatch.setattr("inference.llamacpp_runtime.settings.llamacpp_cache_type_k", "q8_0")
    monkeypatch.setattr("inference.llamacpp_runtime.settings.llamacpp_cache_type_v", "q8_0")

    for context in contexts:
        estimate = estimate_runtime_memory(
            model_bytes=model.stat().st_size,
            metadata=metadata,
            context_tokens=context,
            parallel_sequences=1,
            batch_size=512,
            ubatch_size=128,
            cache_ram_bytes=0,
            prompt_cache_enabled=False,
        )
        if estimate.estimated_bytes + estimate.headroom_bytes > estimate.capacity_bytes:
            continue
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log_path = tmp_path / f"llama-{context}.log"
        started = time.perf_counter()
        with log_path.open("wb") as log_file:
            process = await asyncio.create_subprocess_exec(
                server,
                "--model",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--ctx-size",
                str(context),
                "--threads",
                os.getenv("LLAMACPP_INTEGRATION_THREADS", "8"),
                "--batch-size",
                "512",
                "--ubatch-size",
                "128",
                "--parallel",
                "1",
                "--cache-type-k",
                "q8_0",
                "--cache-type-v",
                "q8_0",
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            rss_task = asyncio.create_task(_sample_peak_rss(process.pid))
            peak_rss = 0
            endpoint = f"http://127.0.0.1:{port}"
            startup_seconds = None
            prompt_tps = generation_tps = ttft_seconds = None
            exit_status = None
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(1800), trust_env=False
                ) as client:
                    deadline = time.perf_counter() + 900
                    while time.perf_counter() < deadline:
                        if process.returncode is not None:
                            break
                        try:
                            health = await client.get(f"{endpoint}/health", timeout=2)
                            if health.status_code == 200:
                                startup_seconds = time.perf_counter() - started
                                break
                        except httpx.HTTPError:
                            pass
                        try:
                            peak_rss = max(peak_rss, psutil.Process(process.pid).memory_info().rss)
                        except psutil.Error:
                            pass
                        await asyncio.sleep(0.5)
                    if startup_seconds is None:
                        raise RuntimeError(f"llama-server failed to become healthy; log={log_path}")

                    target_prompt = max(512, context - 256)
                    prompt = "hello " * target_prompt
                    token_response = await client.post(
                        f"{endpoint}/tokenize", json={"content": prompt, "add_special": True}
                    )
                    token_response.raise_for_status()
                    input_tokens = len(token_response.json().get("tokens", []))
                    if input_tokens + 32 > context:
                        pytest.skip(f"tokenized prompt does not fit {context} context")

                    request_started = time.perf_counter()
                    async with client.stream(
                        "POST",
                        f"{endpoint}/v1/chat/completions",
                        json={
                            "model": model.stem,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 32,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                    ) as response:
                        response.raise_for_status()
                        first_data = True
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            value = line[6:]
                            if value == "[DONE]":
                                continue
                            chunk = json.loads(value)
                            if first_data and chunk.get("choices"):
                                ttft_seconds = time.perf_counter() - request_started
                                first_data = False
                            timings = chunk.get("timings") or {}
                            prompt_tps = timings.get("prompt_per_second", prompt_tps)
                            generation_tps = timings.get("predicted_per_second", generation_tps)
                exit_status = process.returncode
                try:
                    peak_rss = max(peak_rss, psutil.Process(process.pid).memory_info().rss)
                except psutil.Error:
                    pass
            finally:
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, 15)
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(process.wait(), timeout=30)
                    except TimeoutError:
                        try:
                            os.killpg(process.pid, 9)
                        except ProcessLookupError:
                            pass
                        await process.wait()
                exit_status = process.returncode
                rss_task.cancel()
                try:
                    peak_rss = max(peak_rss, await rss_task)
                except asyncio.CancelledError:
                    pass
        results.append(
            {
                "model_sha256": digest,
                "llama_cpp_version_and_build": build,
                "effective_context": context,
                "startup_seconds": startup_seconds,
                "peak_rss_bytes": peak_rss,
                "prompt_tokens": input_tokens,
                "prompt_tokens_per_second": prompt_tps,
                "generation_tokens_per_second": generation_tps,
                "time_to_first_token_seconds": ttft_seconds,
                "exit_status": exit_status,
            }
        )

    if not results:
        pytest.skip("no requested context passed conservative local memory feasibility checks")
    artifact = tmp_path / "qwen3next-integration-results.json"
    artifact.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Qwen3-Next validation artifact: {artifact}\n{json.dumps(results, indent=2)}")
    assert all(item["exit_status"] == 0 for item in results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


async def _sample_peak_rss(pid: int) -> int:
    peak = 0
    while True:
        try:
            peak = max(peak, psutil.Process(pid).memory_info().rss)
        except psutil.Error:
            return peak
        await asyncio.sleep(0.2)
