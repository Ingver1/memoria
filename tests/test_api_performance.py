"""Performance benchmarks for critical API endpoints."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

# Require the real pytest-benchmark plugin for these tests
pytest.importorskip("pytest_benchmark")

from fastapi import FastAPI
from pytest_benchmark.fixture import BenchmarkFixture

try:  # pragma: no cover - optional dependency
    from memory_system.api.app import create_app
except ModuleNotFoundError as exc:  # pragma: no cover - skip if deps missing
    pytest.skip(f"optional dependency not available: {exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Thresholds (in seconds and requests per second)
# ---------------------------------------------------------------------------
MAX_MEAN_TIME = float(os.getenv("MAX_API_MEAN_TIME", "0.5"))
MIN_RPS = float(os.getenv("MIN_API_RPS", "5"))


@pytest.fixture(scope="module")
def app() -> FastAPI:
    """Create a FastAPI application for benchmarking."""
    return create_app()


def test_post_memories_performance(app: FastAPI, benchmark: BenchmarkFixture) -> None:
    """Benchmark POST /api/v1/memory/ for response time and throughput."""
    transport = httpx.ASGITransport(app=app)

    async def _post() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payload: dict[str, Any] = {
                "text": "hello benchmark",
                "modality": "text",
                "user_id": "perf",
            }
            response = await client.post("/api/v1/memory/", json=payload)
            assert response.status_code == 201

    from memory_system.utils.loop import get_loop

    result = benchmark(lambda: get_loop().run_until_complete(_post()))
    stats = result.stats.stats
    assert stats["mean"] < MAX_MEAN_TIME
    assert stats["ops"] > MIN_RPS

    transport.close()


def test_post_search_performance(app: FastAPI, benchmark: BenchmarkFixture) -> None:
    """Benchmark POST /api/v1/memory/search for response time and throughput."""
    transport = httpx.ASGITransport(app=app)

    async def _setup() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/memory/",
                json={"text": "search target", "modality": "text", "user_id": "perf"},
            )

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(_setup())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")

    async def _search() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory/search",
                json={"query": "search target", "top_k": 1},
            )
            assert response.status_code == 200

    from memory_system.utils.loop import get_loop

    result = benchmark(lambda: get_loop().run_until_complete(_search()))
    stats = result.stats.stats
    assert stats["mean"] < MAX_MEAN_TIME
    assert stats["ops"] > MIN_RPS

    transport.close()

