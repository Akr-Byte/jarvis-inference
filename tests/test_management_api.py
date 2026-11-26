import httpx
import pytest

from inference.config import settings
from inference.main import app


@pytest.mark.asyncio
async def test_management_routes_authenticate_and_report_runtime(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "admin-secret")
    monkeypatch.setattr(settings, "allow_insecure_no_auth", False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            denied = await client.get("/v1/runtime/status")
            allowed = await client.get(
                "/v1/runtime/status", headers={"Authorization": "Bearer admin-secret"}
            )
            registry = await client.get(
                "/v1/model-registry", headers={"Authorization": "Bearer admin-secret"}
            )
            metadata = await client.get(
                "/v1/models/qwen3:1.7b",
                headers={"Authorization": "Bearer admin-secret"},
            )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert "state" in allowed.json()["runtime"]
    assert "discovered_models" in registry.json()
    assert metadata.status_code == 200
    assert metadata.json()["registered"] is True


@pytest.mark.asyncio
async def test_external_backend_rejects_managed_model_lifecycle(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "admin-secret")
    monkeypatch.setattr(settings, "llamacpp_managed", False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/models/qwen3:1.7b/load",
                headers={"Authorization": "Bearer admin-secret"},
            )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "LIFECYCLE_UNSUPPORTED"


@pytest.mark.asyncio
async def test_missing_model_file_is_rejected_before_unloading_current_model(monkeypatch) -> None:
    """A bad profile must not cost the operator the model that is already loaded."""
    monkeypatch.setattr(settings, "api_key", "admin-secret")
    monkeypatch.setattr(settings, "llamacpp_managed", True)
    monkeypatch.setattr(settings, "llamacpp_alias", "missing-model")
    monkeypatch.setattr(settings, "llamacpp_model", "/nonexistent/default.gguf")
    monkeypatch.setattr(settings, "llamacpp_model_map_raw", "missing-model=/nonexistent/model.gguf")
    monkeypatch.setattr(settings, "llamacpp_url", "http://127.0.0.1:8081")
    monkeypatch.setenv("MODEL_REGISTRY", "missing-model")
    started: list[object] = []

    async def no_start(*args, **kwargs):
        started.append(args)

    async def no_stop(*args, **kwargs):
        return None

    monkeypatch.setattr("inference.main.runtime.start", no_start)
    monkeypatch.setattr("inference.main.runtime.stop", no_stop)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/models/missing-model/load",
                headers={"Authorization": "Bearer admin-secret"},
            )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "MODEL_FILE_MISSING"
    # The lifespan startup may call start() with no model id; the rejected
    # lifecycle request must not have started any model profile.
    assert all(call == () for call in started)


@pytest.mark.asyncio
async def test_chat_schema_accepts_context_above_32k(monkeypatch) -> None:
    monkeypatch.setattr(settings, "allow_insecure_no_auth", True)
    monkeypatch.setattr(settings, "max_context_length", 131072)

    async def fake_chat(payload, request_id=None):
        return {"choices": [], "usage": {}, "model": payload["model"]}

    monkeypatch.setattr("inference.api.scheduler.chat", fake_chat)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": settings.default_model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "num_ctx": 131072,
                    "max_tokens": 8192,
                },
            )

    assert response.status_code == 200
