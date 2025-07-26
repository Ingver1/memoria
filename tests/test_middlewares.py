from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient, _TestResponse
from memory_system.api.middleware import MaintenanceModeMiddleware, RateLimitingMiddleware
from starlette.responses import JSONResponse, Response


def create_app() -> FastAPI:
    app = FastAPI()
    app.state.maintenance = MaintenanceModeMiddleware(app)
    app.state.rate = RateLimitingMiddleware(app, max_requests=2, window_seconds=60)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _patch_client(client: TestClient) -> None:
    async def _build_response(handler: callable, req: Request) -> Response:
        sig = getattr(handler, "__signature__", None) or inspect.signature(handler)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "request":
                kwargs[name] = req
            elif param.default is not param.empty:
                kwargs[name] = param.default
            else:
                kwargs[name] = None
        result = handler(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, Response):
            return result
        return JSONResponse(result)

    def wrapped_get(url: str, *, params=None, headers=None) -> _TestResponse:
        handler = client._resolve_handler("GET", url)
        if handler is None:
            return _TestResponse(Response(status_code=404))

        req = Request()
        req.app = client.app
        req.headers = headers or {}
        req.client = SimpleNamespace(host="test")
        req.url = SimpleNamespace(path=url)

        async def call_chain(r: Request) -> Response:
            async def final(req: Request) -> Response:
                return await _build_response(handler, req)

            return await client.app.state.rate.dispatch(
                r, lambda rr: client.app.state.maintenance.dispatch(rr, final)
            )

        resp = client._loop.run_until_complete(call_chain(req))
        return _TestResponse(resp)

    client.get = wrapped_get


def test_rate_limit_exceeded() -> None:
    app = create_app()
    with TestClient(app) as client:
        _patch_client(client)

        client.get("/ping")
        client.get("/ping")
        resp = client.get("/ping")

        assert resp.status_code == 429


def test_maintenance_mode_blocks() -> None:
    app = create_app()
    with TestClient(app) as client:
        _patch_client(client)

        resp = client.get("/ping")
        assert resp.status_code == 200

        app.state.maintenance.enable()

        resp = client.get("/ping")
        assert resp.status_code == 503
      
