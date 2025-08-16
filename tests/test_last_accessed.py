import asyncio
import contextlib

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.unified_memory import add, reinforce, update


@pytest.mark.asyncio
async def test_add_writes_last_accessed(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = await add("hi", store=store)
    loaded = await store.get(mem.memory_id)
    with contextlib.suppress(asyncio.CancelledError):
        await store.aclose()
    assert loaded is not None
    assert "last_accessed" in (loaded.metadata or {})


@pytest.mark.asyncio
async def test_update_writes_last_accessed(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = Memory.new("hi")
    await store.add(mem)
    await update(mem.id, text="hello", store=store)
    loaded = await store.get(mem.id)
    with contextlib.suppress(asyncio.CancelledError):
        await store.aclose()
    assert loaded is not None
    assert "last_accessed" in (loaded.metadata or {})


@pytest.mark.asyncio
async def test_reinforce_writes_last_accessed(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    mem = Memory.new("hi")
    await store.add(mem)
    await reinforce(mem.id, 0.2, store=store)
    loaded = await store.get(mem.id)
    with contextlib.suppress(asyncio.CancelledError):
        await store.aclose()
    assert loaded is not None
    assert "last_accessed" in (loaded.metadata or {})
