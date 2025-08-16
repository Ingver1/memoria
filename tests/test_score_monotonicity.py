"""Property-style tests for the ranking score."""

import datetime as dt
import random

import memory_system.unified_memory as um


def test_score_monotonic_in_importance() -> None:
    """Increasing importance should increase the score."""
    now = dt.datetime.now(dt.UTC)
    weights = um.ListBestWeights()
    for _ in range(100):
        base = random.uniform(0.0, 5.0)
        delta = random.uniform(0.1, 5.0)
        meta_low = {"last_accessed": now.isoformat(), "ema_access": 0.0}
        meta_high = {"last_accessed": now.isoformat(), "ema_access": 0.0}
        low = um.Memory("low", "t", now, importance=base, metadata=meta_low)
        high = um.Memory("high", "t", now, importance=base + delta, metadata=meta_high)
        assert um._score_best(high, weights) > um._score_best(low, weights)


def test_score_decreases_with_age() -> None:
    """Older memories (larger Δt) should yield lower scores."""
    now = dt.datetime.now(dt.UTC)
    weights = um.ListBestWeights()
    for _ in range(100):
        base = random.uniform(0.0, 5.0)
        extra = random.uniform(0.1, 5.0)
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
