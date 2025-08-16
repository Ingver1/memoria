"""Hypothesis-based tests for REST API concurrency."""

import asyncio

import pytest

pytest.importorskip("hypothesis")
import httpx
from hypothesis import HealthCheck, given, settings, strategies as st

from memory_system.api.app import create_app
from memory_system.settings import UnifiedSettings

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.needs_fastapi,
    pytest.mark.needs_httpx,
    pytest.mark.needs_hypothesis,
]


@given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
async def test_concurrent_create_and_list(texts: list[str]) -> None:
    """Concurrently create and list memories via the REST API."""
    app = create_app(UnifiedSettings.for_testing())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def create_memory(text: str) -> None:
            resp = await client.post("/api/v1/memory", json={"text": text})
            assert resp.status_code == 201

        async def list_loop() -> None:
            for _ in range(len(texts)):
                resp = await client.get("/api/v1/memory")
                assert resp.status_code == 200

        await asyncio.gather(*(create_memory(t) for t in texts), list_loop())
        final = await client.get("/api/v1/memory")
        assert final.status_code == 200
        data = final.json()
        assert len(data) >= len(texts)
