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
    assert await backend.loaded_models() == []


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
