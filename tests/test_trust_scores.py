import datetime as dt

import pytest

from memory_system.unified_memory import Memory, update_trust_scores


class DummyStore:
    def __init__(self, mems):
        self.memories = {m.memory_id: m for m in mems}

    async def update_memory(self, memory_id, *, metadata=None, **kwargs):
        mem = self.memories[memory_id]
        if metadata:
            if mem.metadata is None:
                mem.metadata = {}
            mem.metadata.update(metadata)
        return mem

    async def upsert_scores(self, scores):
        pass


@pytest.mark.asyncio
async def test_update_trust_scores_accumulates() -> None:
    mem1 = Memory(
        memory_id="1", text="a", created_at=dt.datetime.now(dt.UTC), metadata={"trust_score": 0.0}
    )
    mem2 = Memory(
        memory_id="2", text="b", created_at=dt.datetime.now(dt.UTC), metadata={"trust_score": 0.5}
    )
    st = DummyStore([mem1, mem2])

    async def reason(task, memories):
        return float(len(memories))

    await update_trust_scores("task", [mem1, mem2], reasoner=reason, store=st)
    assert st.memories["1"].metadata["trust_score"] == pytest.approx(1.0)
    assert st.memories["2"].metadata["trust_score"] == pytest.approx(1.5)
