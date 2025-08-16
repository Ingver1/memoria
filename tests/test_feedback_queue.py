import datetime as dt
import random

import pytest

from memory_system.unified_memory import (
    Memory,
    _feedback_heap,
    record_search_feedback,
    sample_feedback,
)


class DummyStore:
    def __init__(self, memories):
        self.memories = memories

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


@pytest.fixture(autouse=True)
def clear_queue():
    _feedback_heap.clear()


@pytest.mark.asyncio
async def test_feedback_queue_sampling():
    random.seed(0)
    now = dt.datetime.now(dt.UTC)
    mem_err = Memory("m1", "foo", now, metadata={"success_count": 0, "trial_count": 0})
    mem_ok = Memory("m2", "foo", now, metadata={"success_count": 10, "trial_count": 10})
    store = DummyStore([mem_err, mem_ok])
    await record_search_feedback(mem_err, False, store=store)
    await record_search_feedback(mem_ok, True, store=store)
    chosen = sample_feedback()[0]
    assert chosen.memory_id == "m1"
    assert len(_feedback_heap) == 1
