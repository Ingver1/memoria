from collections.abc import Sequence

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.asyncio, pytest.mark.needs_hypothesis]
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.store import Memory, SQLiteMemoryStore


# Helper used by property tests
async def _add_with_vectors(store, index, texts, *, importance=None, embed):
    importance = importance or [0.0] * len(texts)
    mems = []
    for text, imp in zip(texts, importance, strict=False):
        mem = Memory.new(text, importance=float(imp))
        await store.add(mem)
        vec = embed(text)
        if getattr(vec, "ndim", 1) == 1:
            vec = np.asarray([vec], dtype=np.float32)
        index.add_vectors([mem.id], vec.astype(np.float32, copy=False))
        mems.append(mem)
    return mems


# Helper to add memories with pre-computed vectors
async def _add_with_vectors(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    texts: Sequence[str],
    *,
    importance: Sequence[float] | None = None,
    embed,
) -> list[Memory]:
    if importance is None:
        importance = [0.0] * len(texts)
    mems: list[Memory] = []
    for text, imp in zip(texts, importance, strict=True):
        mem = Memory.new(text, importance=imp)
        await store.add(mem)
        vec = embed(text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([mem.id], vec.astype(np.float32))
        mems.append(mem)
    return mems


async def _reset_store_index(store: SQLiteMemoryStore, index: FaissHNSWIndex) -> None:
    """Ensure the store and index are empty for property-based tests."""
    existing = await store.search(limit=1000)
    for m in existing:
        await store.delete_memory(m.id)
    index.remove_ids(list(index._reverse_id_map.keys()))


# Property-based test for consolidation on random data
@given(st.lists(st.text(), min_size=2, max_size=10))
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_property_consolidation(store, index, fake_embed, texts: list[str]):
    await _reset_store_index(store, index)
    # Add them to the store and index
    mems = await _add_with_vectors(store, index, texts, embed=fake_embed)

    created = await consolidate_store(store, index, threshold=0.8)
    # Ensure that consolidation occurs, meaning there is at least one summary created
    assert len(created) >= 1  # Check we have at least one summary memory

    # Ensure the originals are removed and summaries are added
    for m in mems:
        assert await store.get(m.id) is None
        assert index.get_vector(m.id) is None


async def test_concat_strategy(store, index, fake_embed):
    mem1 = Memory.new(
        "a cat",
        importance=0.2,
        valence=0.4,
        emotional_intensity=0.6,
        metadata={"trust_score": 0.8},
    )
    mem2 = Memory.new(
        "the cat",
        importance=0.4,
        valence=-0.2,
        emotional_intensity=0.2,
        metadata={"trust_score": 0.4},
    )
    for m in (mem1, mem2):
        await store.add(m)
        vec = fake_embed(m.text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([m.id], vec.astype(np.float32))

    created = await consolidate_store(store, index, threshold=0.8, strategy="concat")

    assert len(created) == 1
    summary = created[0]
    assert summary.text == "a cat the cat"
    assert summary.importance == pytest.approx(np.mean([mem1.importance, mem2.importance]))
    assert summary.valence == pytest.approx(np.mean([mem1.valence, mem2.valence]))
    assert summary.emotional_intensity == pytest.approx(
        np.mean([mem1.emotional_intensity, mem2.emotional_intensity])
    )
    assert summary.metadata.get("trust_score") == pytest.approx(np.mean([0.8, 0.4]))


# Property-based test for forgetting low-scored memories
@given(
    st.lists(
        st.tuples(st.text(), st.floats(min_value=0.0, max_value=1.0)),
        min_size=5,
        max_size=20,
    )
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_property_forgetting(store, index, fake_embed, data: list[tuple[str, float]]):
    await _reset_store_index(store, index)
    # Create memories with random importance scores
    texts, importances = zip(*data, strict=True)
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


async def test_negative_valence_increases_forgetting(store, index, fake_embed):
    """Memories with negative valence should be forgotten before positive ones."""
    pos = Memory.new("pleasant", importance=0.5, valence=0.5, emotional_intensity=0.5)
    neg = Memory.new("unpleasant", importance=0.5, valence=-0.5, emotional_intensity=0.5)

    for mem in (pos, neg):
        await store.add(mem)
        vec = fake_embed(mem.text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([mem.id], vec.astype(np.float32))

    deleted = await forget_old_memories(store, index, min_total=0, retain_fraction=0.5)
    assert deleted == 1

    remaining = {m.id for m in await store.search(limit=10)}
    assert pos.id in remaining
    assert neg.id not in remaining
