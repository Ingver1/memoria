import asyncio
from collections.abc import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memory_system.api.routes.memory import router as memory_router
from memory_system.core.store import lifespan_context
from memory_system.settings import UnifiedSettings

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    app = FastAPI(lifespan=lifespan_context)
    app.include_router(memory_router, prefix="/api/v1/memory")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def test_create_memory_empty_text(test_client: TestClient) -> None:
    response = test_client.post("/api/v1/memory/", json={"text": ""})
    assert response.status_code == 422


def test_create_memory_text_too_long(test_client: TestClient) -> None:
    settings = UnifiedSettings.for_testing()
    long_text = "x" * (settings.security.max_text_length + 1)
    response = test_client.post("/api/v1/memory/", json={"text": long_text})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_concurrent_memory_posts(async_client: httpx.AsyncClient) -> None:
    tasks = [async_client.post("/api/v1/memory/", json={"text": f"mem {i}"}) for i in range(5)]
    responses = await asyncio.gather(*tasks)
    assert all(r.status_code == 201 for r in responses)
