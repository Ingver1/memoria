import datetime as dt

from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


def test_access_frequency_boost() -> None:
    now = dt.datetime.now(dt.UTC)
    mem_low = Memory(
        id="1",
        text="t",
        created_at=now,
        importance=0.5,
        valence=0.0,
        emotional_intensity=0.0,
        metadata={"ema_access": 0.0},
    )
    mem_high = Memory(
        id="2",
        text="t",
        created_at=now,
        importance=0.5,
        valence=0.0,
        emotional_intensity=0.0,
        metadata={"ema_access": 5.0},
    )
    dyn = MemoryDynamics()
    score_low = dyn.score(mem_low, now=now)
    score_high = dyn.score(mem_high, now=now)
    assert score_high > score_low
