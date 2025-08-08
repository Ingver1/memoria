from hypothesis import given, strategies as st
import numpy as np
import pytest

from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.store import Memory
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import SQLiteMemoryStore
from hypothesis.extra import numpy as npst

pytestmark = pytest.mark.asyncio

# Fixture for generating random text
@pytest.fixture
def random_texts():
    return [f"Text number {i}" for i in range(10)]

# Property-based test for consolidation on random data
@given(st.lists(st.text(), min_size=2, max_size=10))
async def test_property_consolidation(store, index, random_texts, fake_embed):
    # Random texts from the strategy
    texts = random_texts
    # Add them to the store and index
    mems = await _add_with_vectors(store, index, texts, embed=fake_embed)

    created = await consolidate_store(store, index, threshold=0.8)
    # Ensure that consolidation occurs, meaning there is at least one summary created
    assert len(created) >= 1  # Check we have at least one summary memory

    # Ensure the originals are removed and summaries are added
    for m in mems:
        assert await store.get(m.id) is None
        assert index.get_vector(m.id) is None


# Property-based test for forgetting low-scored memories
@given(
    st.lists(
        st.tuples(st.text(), st.floats(min_value=0.0, max_value=1.0)), min_size=5, max_size=20
    )
)
async def test_property_forgetting(store, index, random_texts, fake_embed, data):
    # Create memories with random importance scores
    texts, importances = zip(*data)
    mems = await _add_with_vectors(store, index, texts, importance=importances, embed=fake_embed)

    # Forget memories with low importance after consolidation
    deleted = await forget_old_memories(store, index, min_total=10, retain_fraction=0.5)
    assert deleted > 0

    # Ensure that at least the most important memories remain
    for m in mems:
        if m.importance == 1.0:
            assert await store.get(m.id) is not None
        else:
            assert await store.get(m.id) is None
