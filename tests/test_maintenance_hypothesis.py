"""
Property-based tests for maintenance utilities.

These tests ensure that consolidation and forgetting keep the FAISS index and
SQLite store consistent while respecting decay scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.needs_hypothesis

from memory_system.core.index import FaissHNSWIndex
from memory_system.core.maintenance import _decay_score, consolidate_store, forget_old_memories
from memory_system.core.store import Memory, SQLiteMemoryStore


async def _add_memories_with_vectors(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    texts: Sequence[str],
    *,
    importances: Sequence[float] | None = None,
    valences: Sequence[float] | None = None,
    embed,
) -> list[Memory]:
    """Add texts to the store and index with precomputed vectors."""
    if importances is None:
        importances = [0.0] * len(texts)
    if valences is None:
        valences = [0.0] * len(texts)
    mems: list[Memory] = []
    for text, imp, val in zip(texts, importances, valences, strict=True):
        mem = Memory.new(text, importance=float(imp), valence=float(val))
        await store.add(mem)
        mems.append(mem)

    vecs = embed(list(texts))
    if isinstance(vecs, np.ndarray) and vecs.ndim == 1:
        vecs = np.asarray([vecs], dtype=np.float32)
    index.add_vectors([m.id for m in mems], vecs.astype(np.float32, copy=False))
    return mems


@pytest.mark.asyncio
@given(st.lists(st.text(), min_size=2, max_size=10))
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_consolidation_keeps_store_and_index_in_sync(
    store: SQLiteMemoryStore, index: FaissHNSWIndex, fake_embed, texts: list[str]
) -> None:
    mems = await _add_memories_with_vectors(store, index, texts, embed=fake_embed)

    created = await consolidate_store(store, index, threshold=0.8)
    assert created, "consolidation should create summary memories"

    ids_in_store = {m.id for m in await store.search(limit=1000)}
    ids_in_index = set(index._reverse_id_map.keys())
    assert ids_in_store == ids_in_index

    for mem in mems:
        assert mem.id not in ids_in_store


@pytest.mark.asyncio
@given(
    st.lists(
        st.tuples(
            st.text(),
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=-1.0, max_value=1.0),
        ),
        min_size=5,
        max_size=20,
    )
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_forgetting_removes_lowest_decay_scores(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    fake_embed,
    data: list[tuple[str, float, float]],
) -> None:
    texts, importances, valences = zip(*data, strict=True)
    mems = await _add_memories_with_vectors(
        store,
        index,
        texts,
        importances=importances,
        valences=valences,
        embed=fake_embed,
    )

    total = len(mems)
    deleted = await forget_old_memories(store, index, min_total=0, retain_fraction=0.5)
    keep_count = int(total * 0.5)
    assert deleted == total - keep_count

    ids_in_store = {m.id for m in await store.search(limit=1000)}
    ids_in_index = set(index._reverse_id_map.keys())
    assert ids_in_store == ids_in_index
    assert len(ids_in_store) == keep_count

    scores = {
        m.id: _decay_score(
            importance=m.importance,
            valence=m.valence,
            emotional_intensity=m.emotional_intensity,
            age_days=0.0,
        )
        for m in mems
    }
    deleted_ids = set(scores) - ids_in_store
    if deleted_ids:
        max_deleted = max(scores[i] for i in deleted_ids)
        min_kept = min(scores[i] for i in ids_in_store)
        assert max_deleted <= min_kept
    assert min(scores.values()) >= 0.0

