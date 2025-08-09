"""Thread-safety tests for the SQLite memory store."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory_system.core.store import Memory, SQLiteMemoryStore


def test_threaded_add_update_search(tmp_path: Path) -> None:
    """Ensure add, update and search work correctly when run in parallel."""

    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"), pool_size=3)
    asyncio.run(store.initialise())

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

    asyncio.run(preload())

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(asyncio.run, add_memories()),
            executor.submit(asyncio.run, update_memories()),
            executor.submit(asyncio.run, search_loop()),
        ]
        for fut in futures:
            fut.result()

    results = asyncio.run(store.search("mem", limit=200))
    assert len(results) >= 50
    updated = asyncio.run(store.search("updated", limit=50))
    assert len(updated) == len(preloaded_ids)
    assert not store._acquired
