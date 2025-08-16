"""Simple cache implementation for Unified Memory System."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Any

from memory_system.utils.metrics import (
    CACHE_HIT_RATE,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
)


class SmartCache:
    """In-memory cache with optional max size and time-to-live (TTL) support."""

    def __init__(
        self, max_size: int = 1000, ttl: int = 300, *, refresh_on_get: bool = True
    ) -> None:
        """
        Initialize the cache.

        Parameters
        ----------
        max_size:
            Maximum number of items to store.
        ttl:
            Time-to-live for each cache entry in seconds.

        """
        self.max_size = max_size
        self.ttl = ttl
        self.refresh_on_get = refresh_on_get
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()
        # Counters for basic statistics
        self._hits = 0
        self._misses = 0
        self._met_hit_rate = CACHE_HIT_RATE
        self._met_hits = CACHE_HITS_TOTAL
        self._met_misses = CACHE_MISSES_TOTAL

    def _update_metrics(self) -> None:
        total = self._hits + self._misses
        self._met_hit_rate.set(self._hits / total if total else 0.0)

    def _purge_expired(self) -> None:
        """
        Remove items whose TTL has expired.

        This internal method expects the caller to hold ``self._lock``. It scans
        access times for entries older than ``self.ttl`` and removes them from
        both the data and access time maps. If ``ttl`` is non-positive, no work
        is performed.
        """
        if self.ttl <= 0:
            return

        current_time = monotonic()
        expired = [
            key
            for key, last_access in list(self._access_times.items())
            if current_time - last_access > self.ttl
        ]
        for key in expired:
            self._data.pop(key, None)
            self._access_times.pop(key, None)

    def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache by key, honoring TTL if set."""
        with self._lock:
            self._purge_expired()
            if key not in self._data:
                self._misses += 1
                self._met_misses.inc()
                self._update_metrics()
                return None
            if self.refresh_on_get:
                self._access_times[key] = monotonic()
            self._data.move_to_end(key)
            self._access_times.move_to_end(key)
            self._hits += 1
            self._met_hits.inc()
            self._update_metrics()
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache under the given key using LRU eviction."""
        with self._lock:
            self._purge_expired()
            self._data[key] = value
            self._access_times[key] = monotonic()
            self._data.move_to_end(key)
            self._access_times.move_to_end(key)
            if len(self._data) > self.max_size:
                oldest_key, _ = self._data.popitem(last=False)
                self._access_times.pop(oldest_key, None)
            self._update_metrics()

    def clear(self) -> None:
        """Clear all items from the cache and reset statistics."""
        with self._lock:
            self._data.clear()
            self._access_times.clear()

            # Reset statistics so subsequent calls reflect fresh usage

            self._hits = 0
            self._misses = 0
            self._update_metrics()

    def purge_expired(self) -> None:
        """Public wrapper to remove expired entries from the cache."""
        with self._lock:
            self._purge_expired()

    def items(self) -> list[tuple[str, Any]]:
        """
        Return cache items sorted by recency.

        Items are ordered from most recent to least recent. Expired entries
        are purged before returning.
        """
        with self._lock:
            self._purge_expired()
            return list(self._data.items())[::-1]

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about the cache."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {"size": len(self._data), "max_size": self.max_size, "hit_rate": hit_rate}
