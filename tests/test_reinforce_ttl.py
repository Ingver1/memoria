import asyncio

import pytest

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


@pytest.mark.asyncio
async def test_reinforce_drops_expired_memory(store):
    mem = Memory.new("old", ttl_seconds=1)
    await store.add_memory(mem)
    dyn = MemoryDynamics(store)
    await asyncio.sleep(1.2)
    with pytest.raises(RuntimeError):
        await dyn.reinforce(mem.id)
    assert await store.search(text_query="old") == []
