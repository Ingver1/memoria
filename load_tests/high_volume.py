"""
High-volume test for add/search/list_best without vector indexing.

This script inserts a large number of randomly generated memories into a
``SQLiteMemoryStore``.  It then performs a text search and retrieves the
top-scoring memories via :func:`list_best`.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import secrets
import string
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.unified_memory import list_best


def _rand_text(length: int = 20) -> str:
    """Return a random ASCII string of ``length`` characters."""
    return "".join(random.choices(string.ascii_letters, k=length))


async def _item_generator(count: int) -> AsyncIterator[Memory]:
    """Yield ``count`` fresh :class:`Memory` objects with random metadata."""
    for _ in range(count):
        yield Memory.new(
            f"mem {_rand_text()}",
            importance=secrets.randbelow(1000) / 1000,
            valence=random.uniform(-1.0, 1.0),
            emotional_intensity=secrets.randbelow(1000) / 1000,
        )


async def run(count: int) -> None:
    """Populate the store and exercise search and best operations."""
    store = SQLiteMemoryStore("file:memory.db?mode=memory&cache=shared", pool_size=5)
    await store.initialise()
    dynamics = MemoryDynamics(store)

    start = time.time()
    async for mem in _item_generator(count):
        await store.add(mem)
        score = dynamics.score(mem)
        await store.upsert_scores([(mem.id, score)])
    add_time = time.time() - start

    start = time.time()
    hits = await store.search("mem", limit=5)
    search_time = time.time() - start

    start = time.time()
    best = await list_best(n=5, store=store)
    best_time = time.time() - start

    print(
        f"Inserted {count} memories in {add_time:.2f}s\n"
        f"Search returned {len(hits)} results in {search_time:.2f}s\n"
        f"Best returned {len(best)} results in {best_time:.2f}s"
    )

    await store.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="High-volume memory operations")
    parser.add_argument("--count", type=int, default=100_000, help="number of memories")
    args = parser.parse_args()
    asyncio.run(run(args.count))


if __name__ == "__main__":
    main()
