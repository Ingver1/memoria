import pytest

import memory_system.unified_memory as um
from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_access_extends_ttl(monkeypatch, tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = Memory.new("temp", ttl_seconds=10)
    await store.add(mem)
    monkeypatch.setattr(um, "TTL_BUMP_PROB", 1.0)
    monkeypatch.setattr(um.random, "random", lambda: 0.0)
    weights = um.ListBestWeights()
    await um._record_accesses([um._ensure_memory(mem)], store, weights)
    loaded = await store.get(mem.id)
    assert loaded.ttl_seconds and loaded.ttl_seconds > 10
    await store.aclose()
