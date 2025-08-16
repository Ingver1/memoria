from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import faiss
import pytest
from fastapi import FastAPI, Request

import memory_system.core.enhanced_store as es
import memory_system.core.faiss_vector_store as fvs
from ltm_bench.scenario import DEFAULT_DIALOGUE, PII_EMAIL
from memory_system.api import app as app_module
from memory_system.api.app import create_app
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import SecurityConfig, UnifiedSettings

if not hasattr(faiss, "METRIC_INNER_PRODUCT"):
    faiss.METRIC_INNER_PRODUCT = 0  # type: ignore[attr-defined]
    faiss.METRIC_L2 = 1  # type: ignore[attr-defined]


class DummyVectorStore:
    def __init__(self, settings) -> None:
        self.vecs: dict[str, list[float]] = {}

    def add(self, ids, vecs, modality="text"):
        for i, v in zip(ids, vecs, strict=False):
            self.vecs[i] = v.tolist() if hasattr(v, "tolist") else v

    def rebuild(self, modality, vecs, ids):  # pragma: no cover - simple rebuild
        for i, v in zip(ids, vecs, strict=False):
            self.vecs[i] = v.tolist() if hasattr(v, "tolist") else v

    def search(self, vec, k=5, modality="text", ef_search=None):
        ids = list(self.vecs.keys())[:k]
        return ids, [0.0] * len(ids)

    def save(self, path) -> None:  # pragma: no cover - noop
        return None

    def delete(self, ids, modality="text"):
        return None

    def flush(self):  # pragma: no cover - noop
        return None

    def close(self):  # pragma: no cover - noop
        return None

    async def aclose(self) -> None:  # pragma: no cover - noop
        return None

    def stats(self):
        return SimpleNamespace(total_vectors=len(self.vecs))


def _make_app(rate_limit: int = 2, log_pii_details: bool = False) -> FastAPI:
    settings = UnifiedSettings(
        security=SecurityConfig(
            api_token="test-token-123",
            rate_limit_per_minute=rate_limit,
            filter_pii=True,
            pii_log_details=log_pii_details,
        )
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


def _run(
    loop: asyncio.AbstractEventLoop,
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
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

    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    async def call() -> None:
        await app(scope, receive, send)

    loop.run_until_complete(app.router.startup())
    loop.run_until_complete(call())
    loop.run_until_complete(app.router.shutdown())

    status = 500
    headers_out: list[tuple[bytes, bytes]] = []
    body_bytes = b""
    for m in messages:
        if m["type"] == "http.response.start":
            status = m["status"]
            headers_out = m.get("headers", [])
        elif m["type"] == "http.response.body":
            body_bytes += m.get("body", b"")

    return SimpleNamespace(
        status_code=status,
        headers={k.decode(): v.decode() for k, v in headers_out},
        content=body_bytes,
    )


@pytest.mark.needs_fastapi
def test_app_rate_limit_enforced(event_loop: asyncio.AbstractEventLoop, monkeypatch) -> None:
    monkeypatch.setattr(fvs, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    app = _make_app(rate_limit=2)
    headers = {"X-API-Token": "tok"}
    _run(event_loop, app, "GET", "/health/live", headers=headers)
    _run(event_loop, app, "GET", "/health/live", headers=headers)
    resp = _run(event_loop, app, "GET", "/health/live", headers=headers)
    assert resp.status_code == 429
    assert resp.headers["Retry-After"]
    assert resp.headers["X-RateLimit-Limit"] == "2"
    assert resp.headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.needs_fastapi
def test_log_sanitization(caplog, event_loop: asyncio.AbstractEventLoop, monkeypatch) -> None:
    monkeypatch.setattr(fvs, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    app = _make_app(rate_limit=100)

    @app.post("/log")
    async def log_body(request: Request) -> dict[str, str]:
        body = await request.json()
        logging.getLogger().info(body["msg"])
        return {"status": "ok"}

    with caplog.at_level(logging.INFO):
        _run(event_loop, app, "POST", "/log", json_body={"msg": DEFAULT_DIALOGUE[2].text})

    assert PII_EMAIL not in caplog.text
    assert "[EMAIL_REDACTED]" in caplog.text


@pytest.mark.needs_fastapi
def test_pii_logging(caplog, event_loop: asyncio.AbstractEventLoop, monkeypatch) -> None:
    monkeypatch.setattr(fvs, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    app = _make_app(rate_limit=100, log_pii_details=True)
    headers = {"X-API-Token": "tok"}
    body = {"text": f"my email is {PII_EMAIL}"}

    with caplog.at_level(logging.INFO):
        _run(event_loop, app, "POST", "/api/v1/memory/", headers=headers, json_body=body)

    assert "Detected 1 PII items" in caplog.text
    assert "email=1" in caplog.text


def test_store_filters_pii(monkeypatch) -> None:
    monkeypatch.setattr(fvs, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)

    async def run() -> None:
        embedding = [0.0, 0.0, 0.0]
        await store.add_memory(text=f"my email is {PII_EMAIL}", embedding=embedding)
        res = await store.semantic_search(vector=embedding, k=1)
        assert PII_EMAIL not in res[0].text
        await store.close()
    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(run())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")


def test_store_filter_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(fvs, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    cfg = UnifiedSettings(
        security=SecurityConfig(
            api_token="test-token-123", rate_limit_per_minute=1000, filter_pii=False
        )
    )
    store = EnhancedMemoryStore(cfg)

    async def run() -> None:
        embedding = [0.0, 0.0, 0.0]
        await store.add_memory(text=f"my email is {PII_EMAIL}", embedding=embedding)
        res = await store.semantic_search(vector=embedding, k=1)
        assert any(PII_EMAIL in m.text for m in res)
        await store.close()

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(run())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
