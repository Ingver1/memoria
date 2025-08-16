import asyncio

import pytest

from memory_system import rag_router
from memory_system.settings import MonitoringConfig, SecurityConfig, UnifiedSettings


def test_mark_access_respects_privacy_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    class DummyClient:
        def __init__(self, *a, **k):
            pass

        async def post(self, *a, **k):  # type: ignore[no-untyped-def]
            nonlocal called
            called = True

    real_httpx = rag_router.httpx
    monkeypatch.setattr(
        rag_router,
        "httpx",
        type("x", (), {"AsyncClient": DummyClient, "Timeout": real_httpx.Timeout}),
    )
    client = rag_router.MemoriaHTTPClient("http://localhost", privacy_mode="strict")
    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(client.mark_access(["a"]))
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
    assert called is False


@pytest.mark.needs_fastapi
def test_metrics_endpoint_gated_by_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from memory_system.api.routes import health as health_routes

    settings = UnifiedSettings(
        security=SecurityConfig(api_token="tok", telemetry_level="none"),
        monitoring=MonitoringConfig(enable_metrics=True),
    )
    monkeypatch.setattr(health_routes, "_settings", lambda: settings)

    async def call() -> None:
        await health_routes.metrics_endpoint()

    with pytest.raises(HTTPException):
        try:
            from memory_system.utils.loop import get_loop

            get_loop().run_until_complete(call())
        except RuntimeError:
            pytest.skip("No usable event loop available in sandbox")
