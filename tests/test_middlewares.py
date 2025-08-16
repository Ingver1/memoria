from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from memory_system.api.middleware import (
    MaintenanceModeMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware,
)


class _TestResponse:
    """Minimal response wrapper for compatibility with older FastAPI."""

    def __init__(self, resp: Response) -> None:
        self.status_code = resp.status_code
        self.headers = resp.headers
        self._resp = resp

    def json(self) -> Any:  # pragma: no cover - used if available
        if hasattr(self._resp, "json"):
            return self._resp.json()
        raise AttributeError("json")


T = TypeVar("T")


pytestmark = pytest.mark.needs_fastapi


def create_app() -> FastAPI:
    app = FastAPI()

    async def app_handler(
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        response = Response(status_code=200)
        await response(scope, receive, send)

    # Create middleware instances and store them in app.state for access in tests
    maintenance = MaintenanceModeMiddleware(app_handler)
    rate_limit = RateLimitingMiddleware(app_handler, max_requests=2, window_seconds=60)
    security = SecurityHeadersMiddleware(app_handler)
    app.state.maintenance = maintenance
    app.state.rate = rate_limit
    app.state.security = security

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _patch_client(client: Any, loop: asyncio.AbstractEventLoop) -> None:
    async def _build_response(handler: Callable[..., Awaitable[Any]], req: Request) -> Response:
        sig = getattr(handler, "__signature__", None) or inspect.signature(handler)
        kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "request":
                kwargs[name] = req
            elif param.default is not param.empty:
                kwargs[name] = param.default
            else:
                kwargs[name] = None
        result = await handler(**kwargs)
        if isinstance(result, Response):
            return result
        return JSONResponse(content=result)

    def _resolve(method: str, path: str) -> Callable[..., Awaitable[Any]] | None:
        if hasattr(client, "_resolve_handler"):
            return cast(
                "Callable[..., Awaitable[Any]] | None", client._resolve_handler(method, path)
            )
        for route in client.app.routes:
            methods = getattr(route, "methods", [])
            if path == getattr(route, "path", None) and method in methods:
                return cast("Callable[..., Awaitable[Any]]", route.endpoint)
        return None

    def wrapped_get(
        url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> _TestResponse:
        handler = _resolve("GET", url)
        if handler is None:
            return _TestResponse(Response(status_code=404))

        # Create scope for test request
        scope = {
            "type": "http",
            "method": "GET",
            "path": url,
            "query_string": b"",
            "headers": [(k.lower().encode(), str(v).encode()) for k, v in (headers or {}).items()],
            "client": ("test", 0),
            "server": ("test", 80),
            "scheme": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "raw_path": url.encode(),
        }
        request = Request(scope)

        async def call_chain(r: Request) -> Response:
            async def final(req: Request) -> Response:
                response = await _build_response(handler, req)
                return response

            result = await client.app.state.security.dispatch(
                r,
                lambda rr: client.app.state.rate.dispatch(
                    rr, lambda rrr: client.app.state.maintenance.dispatch(rrr, final)
                ),
            )
            if not isinstance(result, Response):
                return JSONResponse(content=result)
            return result

        resp = loop.run_until_complete(call_chain(request))
        return _TestResponse(resp)

    # Monkey-patch the get method
    client.get = wrapped_get


def test_rate_limit_exceeded(event_loop: asyncio.AbstractEventLoop) -> None:
    app = create_app()
    client = SimpleNamespace(app=app)
    _patch_client(client, event_loop)
    client.get("/ping")
    client.get("/ping")
    resp = client.get("/ping")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"]
    assert resp.headers["X-RateLimit-Limit"] == "2"
    assert resp.headers["X-RateLimit-Remaining"] == "0"


def test_rate_limit_by_token(event_loop: asyncio.AbstractEventLoop) -> None:
    app = create_app()
    client = SimpleNamespace(app=app)
    _patch_client(client, event_loop)

    headers1 = {"X-API-Token": "alpha"}
    headers2 = {"X-API-Token": "beta"}

    client.get("/ping", headers=headers1)
    client.get("/ping", headers=headers1)
    resp = client.get("/ping", headers=headers1)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"]
    assert resp.headers["X-RateLimit-Limit"] == "2"
    assert resp.headers["X-RateLimit-Remaining"] == "0"

    # Different token should have its own bucket
    resp = client.get("/ping", headers=headers2)
    assert resp.status_code == 200


def test_maintenance_mode_blocks(event_loop: asyncio.AbstractEventLoop) -> None:
    app = create_app()
    client = SimpleNamespace(app=app)
    _patch_client(client, event_loop)

    resp = client.get("/ping")
    assert resp.status_code == 200

    app.state.maintenance.enable()

    resp = client.get("/ping")
    assert resp.status_code == 503


def test_security_headers_added(event_loop: asyncio.AbstractEventLoop) -> None:
    app = create_app()
    client = SimpleNamespace(app=app)
    _patch_client(client, event_loop)

    resp = client.get("/ping")
    assert resp.headers["Content-Security-Policy"] == "default-src 'self'"
    assert resp.headers["Strict-Transport-Security"].startswith("max-age")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert resp.headers["X-Frame-Options"] == "DENY"
