"""Simple cache implementation for Unified Memory System."""

from __future__ import annotations

import time
from typing import Any


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

    def get(self, key: str) -> Any:
        """Retrieve a value from the cache by key, honoring TTL if set."""
        if key not in self._data:
            return None
        if self.ttl > 0:
            age = time.time() - self._timestamps.get(key, 0)
            if age > self.ttl:
                self._data.pop(key, None)
                self._timestamps.pop(key, None)
                return None
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache under the given key. Evict oldest if over max_size."""
        if len(self._data) >= self.max_size:
            # Evict an arbitrary item (FIFO eviction strategy for simplicity)
            oldest_key = next(iter(self._data))
            self._data.pop(oldest_key, None)
            self._timestamps.pop(oldest_key, None)
        self._data[key] = value
        self._timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._data.clear()
        self._timestamps.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get basic statistics about the cache."""
        # A hit rate statistic could be maintained with additional tracking; here we return 0.0 as a placeholder.
        return {"size": len(self._data), "max_size": self.max_size, "hit_rate": 0.0}
