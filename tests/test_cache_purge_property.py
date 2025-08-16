from unittest.mock import patch

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from memory_system.utils.cache import SmartCache

pytestmark = [pytest.mark.property, pytest.mark.needs_hypothesis]


class DummyTimer:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@given(
    ttl=st.integers(min_value=1, max_value=5),
    advance=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=25)
def test_purge_expired_behavior(ttl: int, advance: int) -> None:
    cache = SmartCache(max_size=10, ttl=ttl)
    timer = DummyTimer()
    with patch("memory_system.utils.cache.monotonic", timer.monotonic):
        cache.put("key", "value")
        timer.advance(advance)
        cache.purge_expired()
        result = cache.get("key")
    if advance > ttl:
        assert result is None
    else:
        assert result == "value"
