"""
If the ANN index is missing, the service should degrade gracefully,
reporting unhealthy status and returning 503 on /health.
"""
import pytest

from fastapi.testclient import TestClient
from memory_system.api.app import create_app
from memory_system.config.settings import UnifiedSettings


def test_index_missing_returns_503(monkeypatch):
    cfg = UnifiedSettings.for_testing()
    app = create_app(cfg)

    # Break the singleton store: replace its index with None.
    from memory_system.api.dependencies import get_store

    store = next(get_store())
    monkeypatch.setattr(store, "_index", None)

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["healthy"] is False
