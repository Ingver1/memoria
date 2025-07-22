"""
Property-based tests for EnhancedMemoryStore vector workflow.

The goal: whatever random float32 vector we add must be retrievable
via an exact semantic search; the store must never raise or lose data.
"""
import pytest

pytestmark = pytest.mark.property

import numpy as np
import pytest_asyncio
from hypothesis import given, settings
from hypothesis import strategies as st
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

VECTOR_DIM = UnifiedSettings.for_testing().model.vector_dim


def _float32_arrays():
    """Generate lists of float32 values in the range [0, 1)."""
    return st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=VECTOR_DIM,
        max_size=VECTOR_DIM,
    ).map(np.float32).map(list)


@pytest_asyncio.fixture(scope="function")
async def store():
    s = EnhancedMemoryStore(UnifiedSettings.for_testing())
    yield s
    await s.close()


@given(vec=_float32_arrays())
@settings(max_examples=20)
async def test_roundtrip_vector(store, vec):
    """Adding then searching the same vector must return exactly one hit."""
    await store.add_memory(text="prop-test", embedding=vec)
    hits = await store.semantic_search(vector=vec, k=1)
    assert len(hits) == 1
    assert hits[0].text == "prop-test"
