import datetime as dt
import math

import pytest

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


def test_logarithmic_decay(monkeypatch) -> None:
    now = dt.datetime.now(dt.UTC)
    past = now - dt.timedelta(days=10)
    mem = Memory(
        id="1",
        text="t",
        created_at=now,
        importance=0.0,
        valence=0.0,
        emotional_intensity=1.0,
        metadata={"last_accessed": past.isoformat()},
    )
    monkeypatch.setattr("memory_system.core.memory_dynamics._decay_law", lambda: "logarithmic")
    monkeypatch.setattr("memory_system.core.memory_dynamics._RECENCY_TAU", float("inf"))
    dyn = MemoryDynamics()
    score = dyn.score(mem, now=now)
    expected = 1.0 / (1.0 + math.log1p(10) / 30.0)
    assert score == pytest.approx(expected)
