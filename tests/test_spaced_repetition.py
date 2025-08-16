import datetime as dt

import pytest

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


@pytest.mark.asyncio
async def test_reinforce_updates_last_used_and_decay(store):
    mem = Memory.new("ping", decay=1.0)
    await store.add_memory(mem)
    dyn = MemoryDynamics(store)
    updated = await dyn.reinforce(mem.id)
    assert updated.last_used is not None
    assert updated.decay > 1.0
    assert (dt.datetime.now(dt.UTC) - updated.last_used).total_seconds() < 5
