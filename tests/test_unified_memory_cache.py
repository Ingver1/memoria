import pytest

from memory_system.unified_memory import (
    Memory,
    _get_cache,
    add,
    delete,
    search,
    update,
)


class DummyStore:
    def __init__(self) -> None:
        self.memories: dict[str, Memory] = {}

    async def add_memory(self, memory: Memory) -> None:  # pragma: no cover - simple store
        self.memories[memory.memory_id] = memory

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter: dict | None = None,
        level: int | None = None,
    ) -> list[Memory]:
        return [m for m in self.memories.values() if query in m.text][:k]

    async def delete_memory(self, memory_id: str) -> None:  # pragma: no cover - simple store
        self.memories.pop(memory_id, None)

    async def update_memory(self, memory_id: str, *, text: str | None = None, metadata=None, **_) -> Memory:
        m = self.memories[memory_id]
        if text is not None:
            m.text = text
        return m

    async def list_recent(self, *, n: int = 20) -> list[Memory]:  # pragma: no cover
        return list(self.memories.values())[-n:]

    async def upsert_scores(self, scores):  # pragma: no cover
        return None

    async def top_n_by_score(
        self, n: int, *, level=None, metadata_filter=None, weights=None, ids=None
    ):  # pragma: no cover
        return list(self.memories.values())[:n]


@pytest.mark.asyncio
async def test_search_cache_invalidated_on_update_delete():
    store = DummyStore()
    cache = _get_cache()
    cache.clear()

    mem = await add("hello world", store=store)
    await search("hello", store=store)
    assert cache.get_stats()["size"] == 1

    await update(mem.memory_id, text="hi there", store=store)
    assert cache.get_stats()["size"] == 0

    await search("hi", store=store)
    assert cache.get_stats()["size"] == 1

    await delete(mem.memory_id, store=store)
    assert cache.get_stats()["size"] == 0
