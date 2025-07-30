from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import ClientHelper

from memory_system.api.routes.memory import router as memory_router
from memory_system.core.store import SQLiteMemoryStore, lifespan_context


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI test application."""
    app = FastAPI(lifespan=lifespan_context)
    app.include_router(memory_router, prefix="/api/v1/memory")
    return app


def test_startup_injects_store(app: FastAPI) -> None:
    """Test that the memory store is properly injected on startup."""
    with ClientHelper(app) as client:
        assert isinstance(client.app.state.memory_store, SQLiteMemoryStore)


def test_add_get_search_cycle(app: FastAPI) -> None:
    """Test the complete cycle of adding, getting, and searching memories."""
    with ClientHelper(app) as client:
        payload = {"text": "hello world"}
        resp = client.post("/api/v1/memory/", json=payload)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        resp = client.get("/api/v1/memory", params={"user_id": None})
        assert resp.status_code == 200

        resp = client.post("/api/v1/memory/search", json={"query": "hello", "top_k": 5})
        assert resp.status_code == 200
        assert any(r["id"] == mem_id for r in resp.json())

        bad = client.post("/api/v1/memory/", json={"text": ""})
        assert bad.status_code == 422


def test_pii_redaction(app: FastAPI) -> None:
    """Test that PII is properly redacted from stored memories."""
    with ClientHelper(app) as client:
        payload = {"text": "Email me at user@example.com"}
        resp = client.post("/api/v1/memory/", json=payload)
        assert resp.status_code == 201

        mem_id = resp.json()["id"]
        resp = client.get("/api/v1/memory", params={"user_id": None})
        assert resp.status_code == 200
        stored = next((m for m in resp.json() if m["id"] == mem_id), None)
        assert stored is not None
        assert "user@example.com" not in stored["text"]


def test_best_memories_endpoint(app: FastAPI) -> None:
    """Test retrieving best memories with limit parameter."""
    with ClientHelper(app) as client:
        for i in range(3):
            payload = {"text": f"mem {i}"}
            client.post("/api/v1/memory/", json=payload)

        resp = client.get("/api/v1/memory/best", params={"limit": 2})
        assert resp.status_code == 200
        assert len(resp.json()) == 2


def test_long_text_validation(app: FastAPI) -> None:
    """Test that memory creation with too long text fails with 422."""
    with ClientHelper(app) as client:
        payload = {"text": "x" * 10001}
        resp = client.post("/api/v1/memory/", json=payload)
        assert resp.status_code == 422


def test_search_invalid_top_k(app: FastAPI) -> None:
    """Test that search with invalid top_k parameter returns 422."""
    with ClientHelper(app) as client:
        resp = client.post("/api/v1/memory/search", json={"query": "hi", "top_k": 0})
        assert resp.status_code == 422
