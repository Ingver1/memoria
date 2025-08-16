"""
If the ANN index is missing, the service should degrade gracefully,
reporting unhealthy status and returning 503 on /health.
"""

import pytest
from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from memory_system.api.app import create_app
from memory_system.settings import UnifiedSettings

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


def test_index_missing_returns_503(monkeypatch: MonkeyPatch) -> None:
    """Test that health check returns 503 when ANN index is missing."""
    cfg = UnifiedSettings.for_testing()
    app = create_app(cfg)

    with TestClient(app) as client:
        store = client.app.state.memory_store
        monkeypatch.setattr(store, "_index", None)

        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["healthy"] is False
