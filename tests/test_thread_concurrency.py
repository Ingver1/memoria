"""Thread-safety tests for the SQLite memory store."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.utils.loop import get_loop


@pytest.mark.asyncio
async def test_threaded_add_update_search(tmp_path: Path) -> None:
    """Ensure add, update and search work correctly when run in parallel."""
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"), pool_size=3)
    await store.initialise()

    preloaded_ids: list[str] = []

    async def preload() -> None:
        for i in range(20):
            mem = Memory.new(f"pre {i}")
            preloaded_ids.append(mem.id)
            await store.add(mem)

    async def add_memories() -> None:
        for i in range(50):
            await store.add(Memory.new(f"mem {i}"))

    async def update_memories() -> None:
        for mid in preloaded_ids:
            await store.update_memory(mid, text="updated")

    async def search_loop() -> None:
        for _ in range(50):
            await store.search("mem", limit=1)

    await preload()

    loop = get_loop()

    async def run_in_thread(coro_func) -> None:
        def runner() -> None:
            asyncio.run_coroutine_threadsafe(coro_func(), loop).result()

        await asyncio.to_thread(runner)

    await asyncio.gather(
        run_in_thread(add_memories),
        run_in_thread(update_memories),
        run_in_thread(search_loop),
    )

    results = await store.search("mem", limit=200)
    assert len(results) >= 50
    updated = await store.search("updated", limit=50)
    assert len(updated) == len(preloaded_ids)
    assert not store._acquired
