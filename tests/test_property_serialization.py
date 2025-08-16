"""
Property-based tests for serialization, reinforcement monotonicity,
and semantic search robustness.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

import pytest_asyncio

pytest.importorskip("hypothesis")
np = pytest.importorskip("numpy")
from hypothesis import HealthCheck, given, settings, strategies as st

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings
from memory_system.unified_memory import reinforce
from memory_system.utils.security import EncryptionManager

pytestmark = [
    pytest.mark.property,
    pytest.mark.needs_hypothesis,
    pytest.mark.needs_crypto,
]

EMBEDDING_DIM = UnifiedSettings.for_testing().model.vector_dim


@pytest_asyncio.fixture
def store() -> AsyncGenerator[EnhancedMemoryStore]:
    """Provide an in-memory EnhancedMemoryStore for tests."""

    async def _store() -> AsyncGenerator[EnhancedMemoryStore]:
        async with EnhancedMemoryStore(UnifiedSettings.for_testing()) as s:
            yield s

    return _store()


def _float32_arrays() -> st.SearchStrategy[list[float]]:
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


@given(text=st.text())
@settings(max_examples=25)
def test_encryption_roundtrip(text: str) -> None:
    """EncryptionManager must roundtrip arbitrary text."""
    mgr = EncryptionManager()
    token = mgr.encrypt(text)
    assert mgr.decrypt(token) == text


@given(
    deltas=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_reinforce_monotonic(
    store: EnhancedMemoryStore, fake_embed, deltas: list[float]
) -> None:
    """Repeated reinforcement with positive deltas must be monotonic."""
    vec = fake_embed("base").tolist()
    mem = await store.add_memory(text="base", embedding=vec)
    prev = mem.importance
    for d in deltas:
        mem = await reinforce(mem.id, amount=d, store=store)
        assert mem.importance >= prev
        prev = mem.importance


@given(query=_float32_arrays(), vecs=st.lists(_float32_arrays(), min_size=1, max_size=5))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_semantic_search_random_embeddings(
    store: EnhancedMemoryStore, query: list[float], vecs: list[list[float]]
) -> None:
    """Semantic search should remain stable for arbitrary embeddings."""
    texts = [f"mem-{i}" for i in range(len(vecs))]
    for text, vec in zip(texts, vecs, strict=True):
        await store.add_memory(text=text, embedding=vec)
    results = await store.semantic_search(embedding=query, k=1)
    assert len(results) == 1
    assert results[0].text in texts
