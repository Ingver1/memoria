"""
Tests correct disabling of the `/metrics` endpoint.
If metrics are disabled, the service must return 404/403
and must not output Prometheus formatted data.
"""
import pytest

from fastapi.testclient import TestClient
from memory_system.api.app import create_app
from memory_system.config.settings import UnifiedSettings


@pytest.fixture(scope="session")
def app_no_metrics():
    cfg = UnifiedSettings.for_testing()
    cfg.monitoring.enable_metrics = False  # crucial line disabling metrics
    return create_app(cfg)


@pytest.fixture
def client(app_no_metrics):
    return TestClient(app_no_metrics)


def test_metrics_endpoint_disabled(client):
    resp = client.get("/metrics")
    assert resp.status_code in (404, 403)
    assert b"# HELP" not in resp.content
