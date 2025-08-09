import numpy as np
import pytest

from memory_system.core.hierarchical_summarizer import HierarchicalSummarizer
from memory_system.core.store import Memory

pytestmark = pytest.mark.asyncio


async def test_singleton_clusters_are_marked_final(store, index, fake_embed):
    mem = Memory.new("a lonely cat")
    await store.add(mem)
    vec = fake_embed(mem.text)
    if isinstance(vec, np.ndarray) and vec.ndim == 1:
        vec = vec.reshape(1, -1)
    index.add_vectors([mem.id], vec.astype(np.float32))

    summarizer = HierarchicalSummarizer(store, index)

    level = 0
    for _ in range(5):
        created = await summarizer.build_level(level)
        if not created:
            break
        level += 1

    assert level == 0
    # memory is marked final and no higher level memories exist
    stored = await store.get(mem.id)
    assert stored and stored.metadata.get("final") is True
    assert await store.search(limit=10, level=1) == []
