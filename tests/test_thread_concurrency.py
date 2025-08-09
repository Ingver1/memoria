import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory_system.core.store import Memory, SQLiteMemoryStore


def test_threaded_add_search(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"), pool_size=2)
    asyncio.run(store.initialise())

    async def add_memories() -> None:
        for i in range(50):
            await store.add(Memory.new(f"mem {i}"))

    async def search_loop() -> None:
        for _ in range(50):
            await store.search("mem", limit=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        add_future = executor.submit(asyncio.run, add_memories())
        search_future = executor.submit(asyncio.run, search_loop())
        add_future.result()
        search_future.result()

    results = asyncio.run(store.search("mem", limit=100))
    assert len(results) >= 50
    assert not store._acquired
