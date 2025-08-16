from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from memory_system.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data


def test_memory_endpoints(client: TestClient) -> None:
    missing = client.post("/api/v1/memory/", json={"text": "unused"})
    assert missing.status_code == 404

    add = client.post("/api/v1/memory/add", json={"text": "fastapi"})
    assert add.status_code == 201
    mem_id = add.json()["id"]

    search = client.post("/api/v1/memory/search", json={"query": "fastapi", "top_k": 5})
    assert search.status_code == 200
    results = search.json()
    assert isinstance(results, list)
    assert any(r["id"] == mem_id for r in results)

    detail = client.get(f"/api/v1/memory/{mem_id}")
    assert detail.status_code == 404
