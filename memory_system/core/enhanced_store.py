# enhanced_store.py — Enhanced memory store for Unified Memory System
#
# Version: v{__version__}
"""Enhanced memory store with health checking and statistics."""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from cryptography.fernet import Fernet

__all__ = ["EnhancedMemoryStore", "HealthComponent"]

from memory_system.config.settings import UnifiedSettings
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore


@dataclass
class HealthComponent:
    """Health check result."""

    healthy: bool
    message: str
    uptime: int
    checks: dict[str, bool]


class EnhancedMemoryStore:
    """Enhanced memory store with health checking and stats."""

    def __init__(self, settings: UnifiedSettings) -> None:
        self.settings = settings
        self._start_time = time.time()
        # Underlying storage components
        dsn = settings.get_database_url()
        self._store = SQLiteMemoryStore(dsn)
        self._index = FaissHNSWIndex(dim=settings.model.vector_dim)
        self._memory_count = 0
        self._closed = False

    async def get_health(self) -> HealthComponent:
        """Get health status."""
        uptime = int(time.time() - self._start_time)
        checks: dict[str, bool] = {}

        try:
            await self._store.ping()
            checks["database"] = True
        except Exception:  # pragma: no cover - connection issues
            checks["database"] = False

        try:
            _ = self._index.stats().total_vectors
            checks["index"] = True
        except Exception:  # pragma: no cover - index errors
            checks["index"] = False

        checks.setdefault("embedding_service", True)

        healthy = all(checks.values())
        message = "All systems operational" if healthy else "Degraded"
        return HealthComponent(
            healthy=healthy,
            message=message,
            uptime=uptime,
            checks=checks,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if self._closed:
            raise RuntimeError("store is closed")
        return {
            "total_memories": self._memory_count,
            "index_size": self._index.stats().total_vectors,
            "cache_stats": {"hit_rate": 0.0},
            "buffer_size": 0,
            "uptime_seconds": int(time.time() - self._start_time),
        }
    
    async def close(self) -> None:
        """Close the store."""
        await self._store.aclose()
        self._closed = True

    # ------------------------------------------------------------------
    # Stubs matching the expected public API used by routes/tests
    # ------------------------------------------------------------------
    async def add_memory(
        self,
        *,
        text: str,
        role: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.0,
        valence: float = 0.0,
        emotional_intensity: float = 0.0,
        embedding: list[float],
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> Memory:
        """Add a memory entry to the database and index."""
        ts = created_at if created_at is not None else time.time()
        text_to_store = text
        if self.settings.security.encrypt_at_rest:
            f = Fernet(self.settings.security.encryption_key.encode())
            text_to_store = f.encrypt(text.encode()).decode()
        mem = Memory(
            id=str(uuid.uuid4()),
            text=text_to_store,
            created_at=dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc),
            importance=importance,
            valence=valence,
            emotional_intensity=emotional_intensity,
            metadata={"role": role, "tags": tags or []},
        )
        await self._store.add(mem)
        self._index.add_vectors([mem.id], np.asarray([embedding], dtype=np.float32))
        self._memory_count += 1
        return mem

    async def semantic_search(
        self,
        *,
        vector: list[float],
        k: int = 5,
        include_embeddings: bool = False,
        ef_search: int | None = None,
    ) -> list[Any]:
        ids, _dists = self._index.search(
            np.asarray(vector, dtype=np.float32), k=k, ef_search=ef_search
        )
        results: list[Any] = []
        for _id in ids:
            mem = await self._store.get(_id)
            if mem is None:
                continue
            if include_embeddings:
                results.append((mem, vector))
            else:
                results.append(mem)
        return results

    async def list_memories(self, user_id: str | None = None) -> list[Memory]:
        if user_id:
            return await self._store.search(metadata_filters={"user_id": user_id})
        return await self._store.search(limit=1000)
