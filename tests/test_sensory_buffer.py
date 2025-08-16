import importlib
import time

from memory_system import sensory_buffer


def test_add_and_get_recent() -> None:
    importlib.reload(sensory_buffer)
    sensory_buffer.add_event({"tool": "ls"}, ttl_seconds=5)
    events = sensory_buffer.get_recent()
    assert len(events) == 1
    assert events[0].payload["tool"] == "ls"


def test_event_expiration() -> None:
    importlib.reload(sensory_buffer)
    sensory_buffer.add_event("temp", ttl_seconds=1)
    time.sleep(1.1)
    sensory_buffer.purge_expired()
    events = sensory_buffer.get_recent()
    assert events == []
