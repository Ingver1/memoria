"""
Tests correct disabling of the `/metrics` endpoint.
If metrics are disabled, the service must return 404/403
and must not output Prometheus formatted data.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from memory_system.api.app import create_app
from memory_system.settings import UnifiedSettings
from memory_system.utils.cache import SmartCache
from memory_system.utils.metrics import (
    CACHE_HIT_RATE,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
)

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest.fixture(scope="session")
def app_no_metrics() -> Any:
    cfg = UnifiedSettings.for_testing()
    cfg.monitoring.enable_metrics = False  # crucial line disabling metrics
    return create_app(cfg)


@pytest.fixture
def client(app_no_metrics: Any) -> TestClient:
    return TestClient(app_no_metrics)


def test_metrics_endpoint_disabled(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code in (404, 403)
    assert b"# HELP" not in resp.content


def test_cache_metrics_tracking() -> None:
    cache = SmartCache(max_size=10, ttl=1)
    hits_before = CACHE_HITS_TOTAL._value.get()
    misses_before = CACHE_MISSES_TOTAL._value.get()

    assert cache.get("x") is None
    cache.put("x", 1)
    assert cache.get("x") == 1

    assert CACHE_HITS_TOTAL._value.get() == hits_before + 1
    assert CACHE_MISSES_TOTAL._value.get() == misses_before + 1
    assert CACHE_HIT_RATE._value.get() == cache.get_stats()["hit_rate"]
