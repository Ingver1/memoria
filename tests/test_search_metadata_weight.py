import datetime as dt

import pytest

from memory_system.unified_memory import Memory, search


class DummyStore:
    def __init__(self, memories):
        self.memories = memories

    async def search_memory(self, *, query, k, metadata_filter=None, level=None, context=None):
        return self.memories


@pytest.mark.asyncio
async def test_search_rerank_uses_metadata_weight() -> None:
    now = dt.datetime.now(dt.UTC)
    mem_low = Memory("m1", "foo bar", now, metadata={"weight": 0.1})
    mem_high = Memory("m2", "foo bar", now, metadata={"weight": 5.0})
    store = DummyStore([mem_low, mem_high])
    res1 = await search("foo", reranker="keyword", store=store, mmr_lambda=None)
    res2 = await search("foo", reranker="keyword", store=store, mmr_lambda=None)
    assert [m.memory_id for m in res1][:2] == ["m2", "m1"]
    assert [m.memory_id for m in res1][:2] == [m.memory_id for m in res2][:2]
