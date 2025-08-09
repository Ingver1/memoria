"""Tests for list_best scoring with signed valence."""

import pytest

from memory_system import memory_helpers as mh
from memory_system import unified_memory as um
from memory_system.core.store import Memory as StoreMemory


@pytest.mark.asyncio
async def test_positive_valence_ranked_above_negative(store):
    """Positive memories should outrank negative ones with equal other scores."""
    pos = await um.add("happy", valence=0.5, emotional_intensity=1.0, importance=1.0, store=store)
    neg = await um.add("sad", valence=-0.9, emotional_intensity=1.0, importance=1.0, store=store)

    best = await um.list_best(n=2, store=store)
    assert best[0].memory_id == pos.memory_id
    assert best[1].memory_id == neg.memory_id


@pytest.mark.asyncio
async def test_negative_can_surface_when_important(store):
    """Highly important negative memories can still rank first."""
    pos = await um.add("minor win", valence=0.2, emotional_intensity=0.1, importance=0.1, store=store)
    neg = await um.add("critical failure", valence=-0.1, emotional_intensity=2.0, importance=2.0, store=store)

    best = await um.list_best(n=2, store=store)
    assert best[0].memory_id == neg.memory_id
    assert best[1].memory_id == pos.memory_id


@pytest.mark.asyncio
async def test_custom_weights_change_ranking(store):
    """Custom weights allow tweaking ranking behaviour."""
    pos = await um.add("good", valence=0.5, emotional_intensity=1.0, importance=1.0, store=store)
    neg = await um.add(
        "bad but vital",
        valence=-0.5,
        emotional_intensity=1.0,
        importance=1.4,
        store=store,
    )

    default_best = await um.list_best(n=2, store=store)
    assert default_best[0].memory_id == pos.memory_id

    weights = mh.ListBestWeights(importance=2.0)

    def _score(m):
        val_w = weights.valence_pos if m.valence >= 0 else weights.valence_neg
        return (
            weights.importance * m.importance + weights.emotional_intensity * m.emotional_intensity + val_w * m.valence
        )

    await store.upsert_scores(
        [
            (pos.memory_id, _score(pos)),
            (neg.memory_id, _score(neg)),
        ]
    )
    custom_best = await mh.list_best(n=2, store=store)
    assert custom_best[0].memory_id == neg.memory_id


@pytest.mark.asyncio
async def test_config_weights_change_ranking(monkeypatch, store):
    """Weights from configuration should affect ranking when not passed explicitly."""
    from memory_system.config.settings import RankingConfig, UnifiedSettings

    settings = UnifiedSettings.for_testing()
    settings.ranking = RankingConfig(importance=2.0)
    monkeypatch.setattr("memory_system.config.settings.get_settings", lambda env=None: settings)

    await um.add(
        "good",
        valence=0.5,
        emotional_intensity=1.0,
        importance=1.0,
        store=store,
    )
    neg = await um.add(
        "bad but vital",
        valence=-0.5,
        emotional_intensity=1.0,
        importance=1.4,
        store=store,
    )

    best = await um.list_best(n=2, store=store)
    assert best[0].memory_id == neg.memory_id


@pytest.mark.asyncio
async def test_level_and_metadata_filters(store):
    """`list_best` should respect level and metadata filters."""
    mem0 = StoreMemory(id="m0", text="base", level=0, metadata={"user_id": "u1"})
    mem1 = StoreMemory(id="m1", text="lvl1", level=1, metadata={"user_id": "u2"})
    await store.add_memory(mem0)
    await store.add_memory(mem1)
    await store.upsert_scores([(mem0.id, 0.1), (mem1.id, 0.2)])

    best_lvl1 = await um.list_best(n=5, store=store, level=1)
    assert [m.memory_id for m in best_lvl1] == [mem1.id]

    best_user = await um.list_best(n=5, store=store, metadata_filter={"user_id": "u1"})
    assert [m.memory_id for m in best_user] == [mem0.id]
