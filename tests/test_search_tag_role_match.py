import datetime as dt

import pytest

from memory_system.unified_memory import Memory, search


class DummyStore:
    def __init__(self, memories):
        self.memories = {m.memory_id: m for m in memories}

    async def search_memory(self, *, query, k, metadata_filter=None, level=None, context=None):
        return list(self.memories.values())

    async def update_memory(self, memory_id, metadata):
        mem = self.memories[memory_id]
        current = dict(mem.metadata or {})
        current.update(metadata)
        mem.metadata = current
        return mem

    async def upsert_scores(self, scores):  # pragma: no cover - not used in assertions
        self.scores = scores


@pytest.mark.asyncio
async def test_search_sets_tag_and_role_match() -> None:
    now = dt.datetime.now(dt.UTC)
    mem_match = Memory("m1", "foo", now, metadata={"tags": ["a"], "role": "user"})
    mem_miss = Memory("m2", "foo", now, metadata={"tags": ["b"], "role": "assistant"})
    store = DummyStore([mem_match, mem_miss])
    ctx = {"tags": ["a"], "role": "user"}
    res1 = await search("foo", store=store, context=ctx, mmr_lambda=None)
    res2 = await search("foo", store=store, context=ctx, mmr_lambda=None)
    meta1 = {m.memory_id: m.metadata for m in res1}
    meta2 = {m.memory_id: m.metadata for m in res2}
    assert meta1["m1"].get("tag_match") == 1.0
    assert meta1["m1"].get("role_match") == 1.0
    assert meta1["m2"].get("tag_match") == 0.0
    assert meta1["m2"].get("role_match") == 0.0
    assert [m.memory_id for m in res1] == [m.memory_id for m in res2]
    assert meta1 == meta2
