import time

import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from memory_system.rag_router import Channel, CompositeScorer, CompositeWeights, MemoryDoc

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]


@given(
    base=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    extra=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_score_decreases_with_age(base: float, extra: float) -> None:
    now = time.time()
    recent = MemoryDoc(
        id="r",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now - base,
        access_count=0,
        globalness=1.0,
    )
    older = MemoryDoc(
        id="o",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now - base - extra,
        access_count=0,
        globalness=1.0,
    )
    scorer = CompositeScorer(CompositeWeights())
    sim = [0.5, 0.5]
    scorer.score(docs=[recent, older], sim=sim, is_global_query=True)
    assert recent.score_composite >= older.score_composite


@given(
    base=st.integers(min_value=0, max_value=100),
    extra=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_score_increases_with_frequency(base: int, extra: int) -> None:
    now = time.time()
    low = MemoryDoc(
        id="l",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now,
        access_count=base,
        globalness=1.0,
    )
    high = MemoryDoc(
        id="h",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now,
        access_count=base + extra,
        globalness=1.0,
    )
    scorer = CompositeScorer(CompositeWeights())
    sim = [0.5, 0.5]
    scorer.score(docs=[low, high], sim=sim, is_global_query=True)
    assert high.score_composite >= low.score_composite


@given(
    sim_a=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    sim_b=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_score_follows_similarity(sim_a: float, sim_b: float) -> None:
    assume(sim_a != sim_b)
    now = time.time()
    a = MemoryDoc(
        id="a",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now,
        access_count=0,
        globalness=1.0,
    )
    b = MemoryDoc(
        id="b",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=now,
        access_count=0,
        globalness=1.0,
    )
    scorer = CompositeScorer(CompositeWeights())
    scorer.score(docs=[a, b], sim=[sim_a, sim_b], is_global_query=True)
    if sim_a > sim_b:
        assert a.score_composite >= b.score_composite
    else:
        assert b.score_composite >= a.score_composite
