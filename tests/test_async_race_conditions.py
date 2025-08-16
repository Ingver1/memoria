"""Async race-condition tests for cache and store operations."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.unified_memory import _get_cache, add, search, update

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.needs_hypothesis,
]


class DummyStore:
    def __init__(self) -> None:
        self.memories: dict[str, Memory] = {}

    async def add_memory(self, memory: Memory) -> None:
        self.memories[memory.memory_id] = memory

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        level: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[Memory]:
        return [m for m in self.memories.values() if query in m.text][:k]

    async def delete_memory(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)

    async def update_memory(
        self, memory_id: str, *, text: str | None = None, metadata=None, **_
    ) -> Memory:
        m = self.memories[memory_id]
        if text is not None:
            m.text = text
        return m

    async def list_recent(self, *, n: int = 20) -> list[Memory]:  # pragma: no cover - simple
        return list(self.memories.values())[-n:]

    async def upsert_scores(self, scores):  # pragma: no cover - simple
        return None

    async def top_n_by_score(
        self, n: int, *, level=None, metadata_filter=None, weights=None, ids=None
    ):  # pragma: no cover
        return list(self.memories.values())[:n]


async def _setup_store(tmp_path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"), pool_size=3)
    await store.initialise()
    return store


@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
async def test_async_store_race_conditions(texts: list[str], tmp_path) -> None:
    """Concurrently update and search the SQLite store."""
    store = await _setup_store(tmp_path)
    memories = [Memory.new(t) for t in texts]
    for mem in memories:
        await store.add(mem)

    async def updater(mem: Memory) -> None:
        await store.update_memory(mem.id, text=mem.text + " updated")

    async def searcher() -> None:
        for _ in range(len(memories)):
            await store.search("updated", limit=100)

    await asyncio.gather(*(updater(m) for m in memories), searcher())
    results = await store.search("updated", limit=100)
    assert len(results) == len(memories)


async def test_cache_race_conditions() -> None:
    """Ensure cache remains consistent under concurrent update and search."""
    store = DummyStore()
    cache = _get_cache()
    cache.clear()
    mem = await add("start", store=store)

    async def updater() -> None:
        for i in range(10):
            await update(mem.memory_id, text=f"text {i}", store=store)

    async def searcher() -> None:
        for _ in range(10):
            res = await search("text", store=store)
            assert res == [] or res[0].text.startswith("text")

    await asyncio.gather(updater(), searcher())
    final = await search("text", store=store)
    assert final and final[0].text.startswith("text")
    assert cache.get_stats()["size"] == 1
