"""Tests for list_best scoring with signed valence."""

import pytest

from memory_system import unified_memory as um


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
