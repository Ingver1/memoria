import datetime as dt
import math

import pytest

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


def test_intensity_decay() -> None:
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
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
    dyn = MemoryDynamics()
    score = dyn.score(mem, now=now)
    expected = math.exp(-10 / 30.0)
    assert score == pytest.approx(expected)
