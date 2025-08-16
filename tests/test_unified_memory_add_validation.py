import datetime as dt

import pytest

from memory_system.unified_memory import Memory, add


class DummyStore:
    def __init__(self) -> None:
        self.memories: list[Memory] = []

    async def add_memory(self, memory: Memory) -> None:
        self.memories.append(memory)

    async def upsert_scores(self, scores) -> None:  # pragma: no cover - simple mock
        return None


@pytest.mark.asyncio
async def test_add_rejects_empty_text() -> None:
    store = DummyStore()
    with pytest.raises(ValueError):
        await add("", store=store)


@pytest.mark.asyncio
async def test_add_sets_utc_timestamp() -> None:
    store = DummyStore()
    mem = await add("hello", store=store)
    assert mem.created_at.tzinfo is dt.UTC
    last_accessed = mem.metadata.get("last_accessed") if mem.metadata else None
    assert last_accessed is not None
    parsed = dt.datetime.fromisoformat(last_accessed)
    assert parsed.tzinfo is dt.UTC
