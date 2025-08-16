import pytest

from memory_system.core.store import SQLiteMemoryStore
from memory_system.unified_memory import add


@pytest.mark.asyncio
async def test_search_returns_extended_fields(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    await add("hello world", metadata={"user_id": "u1"}, store=store)
    results = await store.search(text_query="hello")
    assert results
    meta = results[0].metadata or {}
    assert "last_access_ts" in meta
    assert "access_count" in meta
    assert "personalness" in meta and meta["personalness"] == 1.0
    assert "globalness" in meta and meta["globalness"] == 0.0
    await store.aclose()
