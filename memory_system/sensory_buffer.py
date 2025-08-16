"""
Transient sensory buffer backed by :class:`SmartCache`.

The buffer stores raw outputs from tools, screenshots and other metrics that
should not be persisted long term. Entries are kept in memory and evicted
after ``ttl_seconds`` using an LRU strategy.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from memory_system.utils.cache import SmartCache


@dataclass(slots=True)
class SensoryEvent:
    """Container for an individual sensory event."""

    payload: Any
    timestamp: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.UTC))


_BUFFER: SmartCache | None = None
_BUFFER_LOCK = Lock()


def _get_buffer(ttl_seconds: int | None = None) -> SmartCache:
    """Return the module wide :class:`SmartCache` instance."""
    global _BUFFER  # noqa: PLW0603
    if _BUFFER is None:
        with _BUFFER_LOCK:
            if _BUFFER is None:
                ttl = ttl_seconds
                if ttl is None:
                    try:  # lazy import to avoid heavy config dependency
                        from memory_system.settings import get_settings  # noqa: PLC0415

                        ttl = get_settings().cache.ttl_seconds
                    except Exception:  # noqa: BLE001 - settings optional
                        ttl = 300
                _BUFFER = SmartCache(max_size=1000, ttl=ttl)
    return _BUFFER


def add_event(payload: Any, *, ttl_seconds: int | None = None) -> str:
    """Store a new sensory ``payload`` and return its identifier."""
    cache = _get_buffer(ttl_seconds)
    event_id = uuid.uuid4().hex
    cache.put(event_id, SensoryEvent(payload))
    return event_id


def get_recent(n: int = 20) -> list[SensoryEvent]:
    """Return the ``n`` most recent events in reverse chronological order."""
    cache = _get_buffer()
    purge_expired()
    items = cache.items()
    return [kv[1] for kv in items[:n]]


def purge_expired() -> None:
    """Eagerly remove expired entries from the buffer."""
    cache = _get_buffer()
    cache.purge_expired()


__all__ = ["SensoryEvent", "add_event", "get_recent", "purge_expired"]
