import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_add_memory_rejects_negative_ttl(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = Memory.new("neg", ttl_seconds=-1)
    with pytest.raises(ValueError):
        await store.add_memory(mem)
    await store.aclose()


@pytest.mark.asyncio
async def test_update_memory_rejects_negative_ttl(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = Memory.new("base")
    await store.add(mem)
    with pytest.raises(ValueError):
        await store.update_memory(mem.id, ttl_seconds=-5)
    await store.aclose()
