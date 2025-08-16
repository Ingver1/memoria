import datetime as dt

import pytest

from memory_system.unified_memory import Memory, record_search_feedback, search


class DummyStore:
    def __init__(self, memories):
        self.memories = memories

    async def search_memory(self, *, query, k, metadata_filter=None, level=None, context=None):
        return self.memories

    async def update_memory(
        self,
        memory_id,
        *,
        text=None,
        metadata=None,
        importance=None,
        importance_delta=None,
        valence=None,
        valence_delta=None,
        emotional_intensity=None,
        emotional_intensity_delta=None,
        memory_type=None,
        ttl_seconds=None,
        last_used=None,
        success_score=None,
        decay=None,
    ):
        mem = next(m for m in self.memories if m.memory_id == memory_id)
        if metadata:
            meta = dict(mem.metadata or {})
            meta.update(metadata)
            mem.metadata = meta
        return mem


@pytest.mark.asyncio
async def test_search_reranks_with_ucb() -> None:
    now = dt.datetime.now(dt.UTC)
    mem_a = Memory("a", "foo", now, metadata={"success_count": 10, "trial_count": 10})
    mem_b = Memory("b", "foo", now, metadata={"success_count": 0, "trial_count": 0})
    mem_c = Memory("c", "foo", now, metadata={"success_count": 1, "trial_count": 2})
    store = DummyStore([mem_a, mem_b, mem_c])
    res = await search("foo", store=store, bandit="ucb", mmr_lambda=None)
    assert [m.memory_id for m in res][:3] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_record_search_feedback_updates_counts() -> None:
    now = dt.datetime.now(dt.UTC)
    mem = Memory("m1", "bar", now, metadata={"success_count": 1, "trial_count": 2})
    store = DummyStore([mem])
    updated = await record_search_feedback(mem, True, store=store)
    assert updated.metadata["success_count"] == 2
    assert updated.metadata["trial_count"] == 3


@pytest.mark.asyncio
async def test_search_computes_p_win() -> None:
    now = dt.datetime.now(dt.UTC)
    meta = {"success_count": 3, "trial_count": 5, "schema_type": "experience"}
    mem = Memory("m2", "foo", now, metadata=meta)
    store = DummyStore([mem])
    res = await search("foo", store=store, mmr_lambda=None)
    assert abs(res[0].metadata["p_win"] - ((3 + 1) / (5 + 2))) < 1e-6
