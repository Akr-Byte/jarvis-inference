from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from inference.api import router
from inference.config import settings, validate_security_config
from inference.llamacpp_runtime import runtime, validate_runtime_tuning
from inference.scheduler import scheduler


class BodyLimitMiddleware:
    """Reject oversized POST bodies before pydantic validation (GW-5).

    Implemented as pure ASGI rather than BaseHTTPMiddleware: the streaming
    endpoint must stay a single pass-through response, and the base class adds
    a task per request and interferes with client-disconnect propagation.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not str(scope.get("path", "")).startswith("/v1/")
        ):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        # Chunked requests carry no content-length and would bypass the
        # size cap entirely (GW-5); reject them honestly instead of
        # buffering an unbounded body.
        chunked = "chunked" in (headers.get("transfer-encoding") or "").lower()
        if chunked and not headers.get("content-length"):
            await JSONResponse(
                status_code=411,
                content={
                    "error": {
                        "code": "LENGTH_REQUIRED",
                        "message": (
                            "chunked requests are not accepted; send "
                            "content-length so the request size can be "
                            "enforced"
                        ),
                        "retryable": False,
                    }
                },
            )(scope, receive, send)
            return
        try:
            length = int(headers.get("content-length") or 0)
        except ValueError:
            length = 0
        limit = int(settings.max_request_body_mib * 1024 * 1024)
        if length > limit:
            await JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "REQUEST_TOO_LARGE",
                        "message": (
                            f"request body exceeds the {settings.max_request_body_mib:g} MiB limit"
                        ),
                        "retryable": False,
                    }
                },
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_security_config()
    validate_runtime_tuning()
    if settings.llamacpp_managed:
        await runtime.start()
    scheduler.start_sweeper()
    yield
    try:
        await asyncio.wait_for(scheduler.close(), timeout=settings.shutdown_timeout_seconds)
    except TimeoutError:
        pass
    await runtime.stop()


def _app_version() -> str:
    """Track the installed package version instead of a hardcoded drift (GW-9)."""
    try:
        return package_version("jarvis-inference")
    except PackageNotFoundError:
        return "0.0.0.dev0"


app = FastAPI(title="Jarvis Inference", version=_app_version(), lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware)
app.include_router(router)
