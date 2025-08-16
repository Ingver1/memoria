import datetime as dt

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

import memory_system.unified_memory as um

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]


@given(
    base=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_score_monotonic_in_importance(base: float, delta: float) -> None:
    now = dt.datetime.now(dt.UTC)
    weights = um.ListBestWeights()
    meta = {"last_accessed": now.isoformat(), "ema_access": 0.0}
    low = um.Memory("low", "t", now, importance=base, metadata=meta)
    high = um.Memory("high", "t", now, importance=base + delta, metadata=meta)
    assert um._score_best(high, weights) > um._score_best(low, weights)


@given(
    base=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    extra=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_score_decreases_with_age(base: float, extra: float) -> None:
    now = dt.datetime.now(dt.UTC)
    weights = um.ListBestWeights()
    recent_meta = {
        "last_accessed": (now - dt.timedelta(seconds=base)).isoformat(),
        "ema_access": 0.0,
    }
    older_meta = {
        "last_accessed": (now - dt.timedelta(seconds=base + extra)).isoformat(),
        "ema_access": 0.0,
    }
    recent = um.Memory("r", "t", now, metadata=recent_meta)
    older = um.Memory("o", "t", now, metadata=older_meta)
    assert um._score_best(recent, weights) > um._score_best(older, weights)
