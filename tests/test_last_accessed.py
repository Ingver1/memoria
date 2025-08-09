import asyncio

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.unified_memory import add, reinforce, update


def test_add_writes_last_accessed(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        mem = await add("hi", store=store)
        loaded = await store.get(mem.memory_id)
        await store.aclose()
        assert loaded is not None
        assert "last_accessed" in (loaded.metadata or {})

    asyncio.run(_inner())


def test_update_writes_last_accessed(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        mem = Memory.new("hi")
        await store.add(mem)
        await update(mem.id, text="hello", store=store)
        loaded = await store.get(mem.id)
        await store.aclose()
        assert loaded is not None
        assert "last_accessed" in (loaded.metadata or {})

    asyncio.run(_inner())


def test_reinforce_writes_last_accessed(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        mem = Memory.new("hi")
        await store.add(mem)
        await reinforce(mem.id, 0.2, store=store)
        loaded = await store.get(mem.id)
        await store.aclose()
        assert loaded is not None
        assert "last_accessed" in (loaded.metadata or {})

    asyncio.run(_inner())
