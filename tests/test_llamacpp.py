import pytest

from inference.backends.llamacpp import LlamaCppBackend


class Response:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.status_code = 200
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class Client:
    async def get(self, url: str, timeout: float = 0) -> Response:
        return Response({"data": [{"id": "qwen3:1.7b"}]})


@pytest.mark.asyncio
async def test_llamacpp_models(monkeypatch) -> None:
    monkeypatch.setattr("inference.backends.llamacpp.settings.llamacpp_url", "http://llama")
    backend = LlamaCppBackend()
    backend._client = Client()

    assert await backend.available_models() == ["qwen3:1.7b"]
    assert await backend.loaded_models() == ["qwen3:1.7b"]


@pytest.mark.asyncio
async def test_llamacpp_counts_rendered_chat_prompt_when_supported(monkeypatch) -> None:
    class TokenClient:
        request = None

        async def post(self, url, json, timeout):
            self.request = (url, json, timeout)
            return Response({"tokens": 17})

    monkeypatch.setattr("inference.backends.llamacpp.settings.llamacpp_url", "http://llama")
    backend = LlamaCppBackend()
    client = TokenClient()
    backend._client = client

    count = await backend.count_chat_tokens(
        "coder", {"messages": [{"role": "user", "content": "こんにちは"}], "num_ctx": 4096}
    )

    assert count == 17
    assert client.request[0] == "http://llama/v1/chat/completions/input_tokens"
    assert client.request[1]["model"] == "coder"
    assert "num_ctx" not in client.request[1]


@pytest.mark.asyncio
async def test_llamacpp_token_count_falls_back_when_server_route_is_unavailable(
    monkeypatch,
) -> None:
    class OldServerClient:
        calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            import httpx

            request = httpx.Request("POST", "http://llama/v1/chat/completions/input_tokens")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("missing route", request=request, response=response)

    monkeypatch.setattr("inference.backends.llamacpp.settings.llamacpp_url", "http://llama")
    backend = LlamaCppBackend()
    client = OldServerClient()
    backend._client = client

    assert await backend.count_chat_tokens("coder", {"messages": []}) is None
    assert await backend.count_chat_tokens("coder", {"messages": []}) is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_llamacpp_health_is_false_when_not_configured(monkeypatch) -> None:
    monkeypatch.setattr("inference.backends.llamacpp.settings.llamacpp_url", "")
    backend = LlamaCppBackend()

    assert await backend.health() is False
    await backend.close()


@pytest.mark.asyncio
async def test_llamacpp_health_rejects_error_status(monkeypatch) -> None:
    class ErrorResponse:
        status_code = 404

    class ErrorClient:
        async def get(self, url: str, timeout: float = 0) -> ErrorResponse:
            return ErrorResponse()

    monkeypatch.setattr("inference.backends.llamacpp.settings.llamacpp_url", "http://llama")
    backend = LlamaCppBackend()
    backend._client = ErrorClient()

    assert await backend.health() is False


def test_llamacpp_available_models_degrades_on_transport_error(monkeypatch) -> None:
    import asyncio

    import httpx

    from inference.backends.llamacpp import LlamaCppBackend
    from inference.config import settings

    class FailingClient:
        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("backend down")

    backend = LlamaCppBackend()
    backend._client = FailingClient()
    monkeypatch.setattr(settings, "llamacpp_url", "http://llama")

    assert asyncio.run(backend.available_models()) == []
