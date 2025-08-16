import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # FastAPI >= 0.111
    from fastapi.testclient import ClientHelper as TestClient

from memory_system.api.app import create_app
from memory_system.settings import UnifiedSettings

pytestmark = pytest.mark.needs_fastapi


def test_memory_aging_endpoint() -> None:
    app = create_app(UnifiedSettings.for_testing())
    with TestClient(app) as client:
        resp = client.post("/api/v1/memory/", json={"text": "hello"})
        assert resp.status_code == 201
        mem_id = resp.json()["id"]
        resp2 = client.get("/api/v1/memory/aging")
        assert resp2.status_code == 200
        data = resp2.json()
        assert any(item["id"] == mem_id for item in data)
