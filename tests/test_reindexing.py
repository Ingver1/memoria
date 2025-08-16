import asyncio

import pytest
from fastapi.testclient import TestClient

from memory_system.api.app import create_app
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]


@pytest.mark.asyncio
async def test_add_blocks_during_reindexing():
    cfg = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(cfg) as store:
        await store._index_lock.acquire()
        store._reindexing = True

        async def _add():
            await store.add_memory(text="t", embedding=[0.0] * cfg.model.vector_dim)

        task = asyncio.create_task(_add())
        await asyncio.sleep(0.1)
        assert not task.done()
        store._reindexing = False
        store._index_lock.release()
        await asyncio.wait_for(task, timeout=1)


def test_health_reports_reindexing():
    cfg = UnifiedSettings.for_testing()
    app = create_app(cfg)
    with TestClient(app) as client:
        store = client.app.state.memory_store
        store._reindexing = True
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "reindexing"
