"""Simple cache implementation for Unified Memory System."""

from __future__ import annotations

import time
from typing import Any

from memory_system.utils.metrics import (
    CACHE_HIT_RATE,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
)


class SmartCache:
    """In-memory cache with optional max size and time-to-live (TTL) support."""

    def __init__(self, max_size: int = 1000, ttl: int = 300) -> None:
        """Initialize the cache.

        Parameters
        ----------
        max_size:
            Maximum number of items to store.
        ttl:
            Time-to-live for each cache entry in seconds.
        """
        self.max_size = max_size
        self.ttl = ttl
        self._data: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        # Counters for basic statistics
        self._hits = 0
        self._misses = 0
        self._met_hit_rate = CACHE_HIT_RATE
        self._met_hits = CACHE_HITS_TOTAL
        self._met_misses = CACHE_MISSES_TOTAL

    def _update_metrics(self) -> None:
        total = self._hits + self._misses
        self._met_hit_rate.set(self._hits / total if total else 0.0)

    def get(self, key: str) -> Any:
        """Retrieve a value from the cache by key, honoring TTL if set."""
        if key not in self._data:
            self._misses += 1
            self._met_misses.inc()
            self._update_metrics()
            return None
        if self.ttl > 0:
            age = time.time() - self._timestamps.get(key, 0)
            if age > self.ttl:
                self._data.pop(key, None)
                self._timestamps.pop(key, None)
                self._misses += 1
                self._met_misses.inc()
                self._update_metrics()
                return None
        self._hits += 1
        self._met_hits.inc()
        self._update_metrics()
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache under the given key. Evict oldest by timestamp if over max_size."""
        if len(self._data) >= self.max_size:
            # Evict the oldest item by timestamp (LRU eviction)
            oldest_key = min(self._timestamps, key=lambda k: self._timestamps[k])
            self._data.pop(oldest_key, None)
            self._timestamps.pop(oldest_key, None)
        self._data[key] = value
        self._timestamps[key] = time.time()
        self._update_metrics()

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._data.clear()
        self._timestamps.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about the cache."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {"size": len(self._data), "max_size": self.max_size, "hit_rate": hit_rate}
