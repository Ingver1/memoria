from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request

from memory_system.api import app as app_module
from memory_system.api.app import create_app
from memory_system.config.settings import SecurityConfig, UnifiedSettings


def _make_app(rate_limit: int = 2) -> FastAPI:
    settings = UnifiedSettings(
        security=SecurityConfig(api_token="test-token-123", rate_limit_per_minute=rate_limit, filter_pii=True)
    )

    async def _dummy_get_store(path):  # pragma: no cover - simple stub
        class _Store:
            async def ping(self) -> None:
                return None

            async def aclose(self) -> None:
                return None

        return _Store()

    app_module.get_store = _dummy_get_store
    return create_app(settings)


pytestmark = pytest.mark.needs_fastapi


def _run(
    app: FastAPI, method: str, path: str, *, headers: dict[str, str] | None = None, json_body: dict | None = None
) -> SimpleNamespace:
    body = b""
    headers_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers_list.append((b"content-type", b"application/json"))

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": headers_list,
        "client": ("test", 0),
        "server": ("test", 80),
        "scheme": "http",
        "asgi": {"version": "3.0", "spec_version": "2.1"},
        "raw_path": path.encode(),
    }

    async def receive() -> dict[str, bytes | bool]:
        return {"type": "http.request", "body": body, "more_body": False}

    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    async def call() -> None:
        await app(scope, receive, send)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(app.router.startup())
        loop.run_until_complete(call())
    finally:
        loop.run_until_complete(app.router.shutdown())
        loop.close()

    status = 500
    headers_out: list[tuple[bytes, bytes]] = []
    body_bytes = b""
    for m in messages:
        if m["type"] == "http.response.start":
            status = m["status"]
            headers_out = m.get("headers", [])
        elif m["type"] == "http.response.body":
            body_bytes += m.get("body", b"")

    resp = SimpleNamespace(
        status_code=status,
        headers={k.decode(): v.decode() for k, v in headers_out},
        content=body_bytes,
    )
    return resp


def test_app_rate_limit_enforced() -> None:
    app = _make_app(rate_limit=2)
    headers = {"X-API-Token": "tok"}
    _run(app, "GET", "/health/live", headers=headers)
    _run(app, "GET", "/health/live", headers=headers)
    resp = _run(app, "GET", "/health/live", headers=headers)
    assert resp.status_code == 429


def test_log_sanitization(caplog) -> None:
    app = _make_app(rate_limit=100)

    @app.post("/log")
    async def log_body(request: Request) -> dict[str, str]:
        body = await request.json()
        logging.getLogger().info(body["msg"])
        return {"status": "ok"}

    with caplog.at_level(logging.INFO):
        _run(app, "POST", "/log", json_body={"msg": "email test@example.com"})

    assert "test@example.com" not in caplog.text
    assert "[EMAIL_REDACTED]" in caplog.text
