"""
Adds a memory through REST, retrieves it, asserts round-trip integrity.
Uses TestClient instead of HTTPX to stay in-process and fast.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from memory_system.api.routes.memory import router as memory_router
from memory_system.core.store import lifespan_context

app = FastAPI(lifespan=lifespan_context)
app.include_router(memory_router, prefix="/api/v1/memory")


def test_add_and_get_memory():
    """Round-trip memory through the public API."""
    with TestClient(app) as client:
        # 1. add memory
        r = client.post("/api/v1/memory/", json={"text": "e2e-snap"})
        assert r.status_code == 201
        memory_id = r.json()["id"]

        # 2. list
        s = client.get("/api/v1/memory", params={"user_id": None})
        assert s.status_code == 200
        assert any(m["id"] == memory_id for m in s.json())

        # 3. search by text
        q = client.post("/api/v1/memory/search", json={"query": "e2e-snap", "top_k": 1})
        assert q.status_code == 200
        assert any(r["id"] == memory_id for r in q.json())
