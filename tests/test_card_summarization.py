from time import perf_counter

from memory_system.core.store import Memory
from memory_system.core.summarization import head2tail


def test_head2tail_prefers_card_lesson() -> None:
    mem = Memory.new("full text", metadata={"card": {"lesson": "short"}})
    summary = head2tail([mem])
    assert summary == "short"


def test_head2tail_quality_and_speed() -> None:
    mems = [
        Memory.new(
            "primary text",
            importance=0.9,
            metadata={"card": {"lesson": "first"}},
        ),
        Memory.new(
            "secondary text",
            importance=0.5,
            metadata={"card": {"lesson": "second"}},
        ),
        Memory.new("extra", importance=0.1),
    ]

    start = perf_counter()
    summary = head2tail(mems)
    duration = perf_counter() - start

    assert summary == "first … second"
    assert duration < 0.01
