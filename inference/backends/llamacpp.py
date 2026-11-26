from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from inference.config import settings
from inference.errors import InferenceError


class LlamaCppBackend:
    name = "llama.cpp"

    def __init__(self) -> None:
        self._token_count_route_supported: bool | None = None
        timeout = httpx.Timeout(
            settings.backend_http_timeout_seconds,
            connect=min(10.0, settings.backend_http_timeout_seconds),
        )
        limits = httpx.Limits(
            # Permit one connection per llama.cpp slot plus a short-headroom
            # spare, so gateway-level concurrency is never serialized inside
            # the HTTP pool.
            max_connections=max(4, settings.max_concurrent_requests + 2),
            max_keepalive_connections=max(2, settings.max_concurrent_requests),
            keepalive_expiry=30.0,
        )
        self._client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        if not settings.llamacpp_base_url:
            return False
        try:
            response = await self._client.get(f"{settings.llamacpp_base_url}/health", timeout=3)
            return 200 <= response.status_code < 300
        except httpx.HTTPError:
            return False

    async def available_models(self) -> list[str]:
        if not settings.llamacpp_base_url:
            return []
        try:
            response = await self._client.get(f"{settings.llamacpp_base_url}/v1/models", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError:
            # An unreachable backend must degrade to an empty catalog (the
            # same contract as health()), not leak a raw httpx exception
            # through the scheduler readiness path.
            return []
        return [str(item["id"]) for item in response.json().get("data", [])]

    async def loaded_models(self) -> list[str]:
        # llama-server serves exactly one model. Its OpenAI model catalog is
        # the authoritative loaded model identity (including configured alias).
        return await self.available_models()

    async def count_chat_tokens(self, model: str, payload: dict[str, Any]) -> int | None:
        """Ask llama-server to count its rendered chat prompt, when supported."""
        if not settings.llamacpp_base_url:
            return None
        if self._token_count_route_supported is False:
            return None
        body = dict(payload)
        body.pop("num_ctx", None)
        body["model"] = model
        try:
            response = await self._client.post(
                f"{settings.llamacpp_base_url}/v1/chat/completions/input_tokens",
                json=body,
                timeout=3,
            )
            response.raise_for_status()
            data = response.json()
            count = data.get("tokens", data.get("prompt_tokens"))
            self._token_count_route_supported = True
            return int(count) if isinstance(count, int) and count >= 0 else None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {404, 405, 501}:
                self._token_count_route_supported = False
            return None
        except (httpx.HTTPError, ValueError, TypeError):
            # Older llama-server versions do not expose this route. Fall back
            # to the explicitly estimated admission path without failing chat.
            return None

    async def embeddings(self, model: str, inputs: list[str]) -> list[list[float]]:
        if not settings.llamacpp_base_url:
            raise InferenceError(
                "BACKEND_NOT_CONFIGURED", "llama.cpp is not configured", False, 503
            )
        try:
            response = await self._client.post(
                f"{settings.llamacpp_base_url}/v1/embeddings",
                json={"model": model, "input": inputs},
            )
            response.raise_for_status()
            data = response.json()
            return [list(map(float, item["embedding"])) for item in data.get("data", [])]
        except httpx.TimeoutException as exc:
            raise InferenceError(
                "EMBEDDING_TIMEOUT", "Embedding request timed out", True, 504
            ) from exc
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code >= 500
            raise InferenceError(
                "BACKEND_ERROR",
                f"llama.cpp returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                retryable,
                503 if retryable else 400,
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError("BACKEND_UNAVAILABLE", str(exc), True, 503) from exc

    async def chat(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.llamacpp_base_url:
            raise InferenceError(
                "BACKEND_NOT_CONFIGURED", "llama.cpp is not configured", False, 503
            )
        body = dict(payload)
        body.pop("num_ctx", None)  # Runtime context is configured at process launch.
        body["model"] = model
        try:
            response = await self._client.post(
                f"{settings.llamacpp_base_url}/v1/chat/completions", json=body
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise InferenceError("MODEL_TIMEOUT", "Inference request timed out", True, 504) from exc
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code >= 500
            raise InferenceError(
                "BACKEND_ERROR",
                f"llama.cpp returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                retryable,
                503 if retryable else 400,
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError("BACKEND_UNAVAILABLE", str(exc), True, 503) from exc
        return response.json()

    async def stream(self, model: str, payload: dict[str, Any]) -> AsyncIterator[str]:
        if not settings.llamacpp_base_url:
            raise InferenceError(
                "BACKEND_NOT_CONFIGURED", "llama.cpp is not configured", False, 503
            )
        body = dict(payload)
        body.pop("num_ctx", None)
        body["model"] = model
        body["stream"] = True
        stream_options = dict(body.get("stream_options") or {})
        # Request a terminal usage chunk unconditionally: the gateway needs
        # token accounting even when the client did not ask for usage. The API
        # layer suppresses the usage-only chunk for clients that did not opt in.
        stream_options["include_usage"] = True
        body["stream_options"] = stream_options
        try:
            async with self._client.stream(
                "POST", f"{settings.llamacpp_base_url}/v1/chat/completions", json=body
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]
        except httpx.TimeoutException as exc:
            raise InferenceError("MODEL_TIMEOUT", "Inference request timed out", True, 504) from exc
        except httpx.HTTPStatusError as exc:
            retryable = exc.response.status_code >= 500
            raise InferenceError(
                "BACKEND_ERROR",
                f"llama.cpp returned HTTP {exc.response.status_code}: {exc.response.text[:500]}",
                retryable,
                503 if retryable else 400,
            ) from exc
        except httpx.HTTPError as exc:
            raise InferenceError("BACKEND_UNAVAILABLE", str(exc), True, 503) from exc
