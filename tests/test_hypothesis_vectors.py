"""
Property-based tests for EnhancedMemoryStore vector workflow.

The goal: whatever random float32 vector we add must be retrievable
via an exact semantic search; the store must never raise or lose data.
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

try:  # pragma: no cover - optional dependency
    from hypothesis import HealthCheck, given, settings, strategies as st
    from hypothesis.strategies import SearchStrategy
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]

EMBEDDING_DIM = UnifiedSettings.for_testing().model.vector_dim


def _float32_arrays() -> SearchStrategy[list[float]]:
    """Generate lists of float32 values in the range [0, 1)."""
    return (
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=EMBEDDING_DIM,
            max_size=EMBEDDING_DIM,
        )
        .map(np.float32)
        .map(list)
    )


@pytest_asyncio.fixture(scope="function")
async def store() -> AsyncGenerator[EnhancedMemoryStore, None]:
    """Create an EnhancedMemoryStore instance for testing."""
    async with EnhancedMemoryStore(UnifiedSettings.for_testing()) as s:
        yield s


@given(vec=_float32_arrays())
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_roundtrip_embedding(store: EnhancedMemoryStore, vec: list[float]) -> None:
    """Adding then searching the same embedding must return exactly one hit."""
    await store.add_memory(text="prop-test", embedding=vec)
    hits = await store.semantic_search(embedding=vec, k=1)
    assert len(hits) == 1
    assert hits[0].text == "prop-test"
