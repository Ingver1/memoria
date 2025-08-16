import datetime as dt

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st

import memory_system.core.memory_dynamics as md
from memory_system.core.store import Memory

pytestmark = [
    pytest.mark.property,
    pytest.mark.needs_hypothesis,
]


@given(
    base=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_reinforcement_increases_score(store, base: float, delta: float) -> None:
    """Reinforcement should never decrease a memory's score."""
    mem = Memory.new("base", importance=base)
    await store.add_memory(mem)
    dyn = md.MemoryDynamics(store)
    before = dyn.score(mem)
    updated = await dyn.reinforce(mem.id, amount=delta)
    after = dyn.score(updated)
    assert after >= before


@given(
    importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    intensity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    valence=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    age=st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False),
    extra=st.floats(min_value=0.1, max_value=30.0, allow_nan=False, allow_infinity=False),
)
def test_aging_decreases_score(
    importance: float,
    intensity: float,
    valence: float,
    age: float,
    extra: float,
) -> None:
    """Older memories should never score higher than newer ones."""
    dyn = md.MemoryDynamics()
    now = dt.datetime.now(dt.UTC)
    recent_meta = {"last_accessed": (now - dt.timedelta(days=age)).isoformat()}
    older_meta = {"last_accessed": (now - dt.timedelta(days=age + extra)).isoformat()}
    recent = Memory.new(
        "recent",
        importance=importance,
        valence=valence,
        emotional_intensity=intensity,
        metadata=recent_meta,
    )
    older = Memory.new(
        "older",
        importance=importance,
        valence=valence,
        emotional_intensity=intensity,
        metadata=older_meta,
    )
    assert dyn.score(recent, now=now) >= dyn.score(older, now=now)


@given(ttl=st.integers(min_value=1, max_value=5))
@settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_ttl_removes_expired_entries(store, ttl: int, monkeypatch) -> None:
    """Expired memories should be removed when reinforced."""
    mem = Memory.new("temp", ttl_seconds=ttl)
    await store.add_memory(mem)
    dyn = md.MemoryDynamics(store)
    future = mem.created_at + dt.timedelta(seconds=ttl + 1)

    class FrozenDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return future if tz is None else future.astimezone(tz)

    monkeypatch.setattr(md.dt, "datetime", FrozenDateTime)
    with pytest.raises(RuntimeError):
        await dyn.reinforce(mem.id)
    assert await store.search(text_query="temp") == []
