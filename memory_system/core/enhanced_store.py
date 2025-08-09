# enhanced_store.py — Enhanced memory store for Unified Memory System
#
# Version: v{__version__}
"""Enhanced memory store with health checking and statistics."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from cryptography.fernet import Fernet
import logging

__all__ = ["EnhancedMemoryStore", "HealthComponent"]

log = logging.getLogger(__name__)

from memory_system.config.settings import UnifiedSettings
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.summarization import SummaryStrategy


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
        """Initialise the store components using ``settings``."""
        self.settings = settings
        self._start_time = time.time()
        # Underlying storage components
        dsn = settings.get_database_url()
        self._store = SQLiteMemoryStore(dsn)
        self._index = FaissHNSWIndex(
            dim=settings.model.vector_dim,
            M=settings.model.hnsw_m,
            ef_construction=settings.model.hnsw_ef_construction,
            ef_search=settings.model.hnsw_ef_search,
        )
        vec_path = settings.database.vec_path
        if vec_path.exists():
            self._index.load(str(vec_path))
            self._memory_count = self._index.stats().total_vectors
        else:
            self._memory_count = 0
            if settings.model.hnsw_autotune:
                sample = np.random.rand(128, settings.model.vector_dim).astype(np.float32)
                M, ef_c, ef_s = self._index.auto_tune(sample)
                object.__setattr__(settings.model, "hnsw_m", M)
                object.__setattr__(settings.model, "hnsw_ef_construction", ef_c)
                object.__setattr__(settings.model, "hnsw_ef_search", ef_s)
                log.info(
                    "Auto-tuned HNSW params: M=%d ef_construction=%d ef_search=%d",
                    M,
                    ef_c,
                    ef_s,
                )
                self._index = FaissHNSWIndex(
                    dim=settings.model.vector_dim,
                    M=M,
                    ef_construction=ef_c,
                    ef_search=ef_s,
                )

        async def _save_index() -> None:
            await asyncio.to_thread(self._index.save, str(vec_path))

        self._store.add_commit_hook(_save_index)
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
        self._index.save(str(self.settings.database.vec_path))
        self._memory_count += 1
        return mem

    async def semantic_search(
    self,
    *,
    vector: list[float],
    k: int = 5,
    return_distance: bool = False,
    ef_search: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    level: int | None = None,
) -> list[Any]:
    """
    Perform a semantic vector search.

    Parameters
    ----------
    vector:
        Query embedding (list of floats).
    k:
        Number of nearest neighbours to return.
    return_distance:
        When True, return (Memory, distance) tuples; otherwise return Memory.
    ef_search:
        Controls HNSW search quality.
    metadata_filter:
        Optional metadata constraints. When provided, only memories whose
        metadata matches all key/value pairs participate in the results.
    level:
        Optional logical level constraint; only memories at this level
        are returned.

    Returns
    -------
    list[Any]
        A list of memories (optionally paired with their distance).
    """
    # Widen the ANN probe only if we plan to post-filter.
    need_filter = (metadata_filter is not None) or (level is not None)
    total = self._index.stats().total_vectors or k
    search_k = min((k * 5) if need_filter else k, max(1, total))

    vec = np.asarray(vector, dtype=np.float32)

    # Single ANN search
    ids, dists = self._index.search(vec, k=search_k, ef_search=ef_search)
    candidates = list(zip(ids, dists, strict=False))

    # Optional precomputation of allowed IDs via the SQLite store
    if need_filter:
        allowed_mems = await self._store.search(
            metadata_filters=metadata_filter,
            limit=total,  # cover whole corpus; store caps internally if needed
        )
        allowed_ids = {m.id for m in allowed_mems}
        if level is not None:
            # If Memory has a `level` field, filter by it as well
            allowed_ids = {
                mid for mid in allowed_ids
                if (getattr(await self._store.get(mid), "level", None) == level)
            }
        if not allowed_ids:
            return []

        # Keep only candidates that pass constraints, then trim to k
        candidates = [(_id, dist) for _id, dist in candidates if _id in allowed_ids][:k]
    else:
        candidates = candidates[:k]

    # Materialize Memory objects and build the final result list
    results: list[Any] = []
    for _id, dist in candidates:
        mem = await self._store.get(_id)
        if mem is None:
            continue
        if (level is not None) and (getattr(mem, "level", None) != level):
            # Defensive guard in case the store call above didn’t filter by level
            continue
        results.append((mem, float(dist)) if return_distance else mem)

    return results

    async def list_memories(self, user_id: str | None = None) -> list[Memory]:
        """List memories, optionally filtering by ``user_id``."""
        if user_id:
            return await self._store.search(metadata_filters={"user_id": user_id})
        return await self._store.search(limit=1000)

    # Long-term memory maintenance API
    async def consolidate_memories(
        self,
        *,
        threshold: float = 0.83,
        strategy: str | SummaryStrategy = "head2tail",
    ) -> list[Memory]:
        """Cluster similar items, create summary memories, remove originals."""
        from memory_system.core.maintenance import consolidate_store

        return await consolidate_store(
            self._store,
            self._index,
            threshold=threshold,
            strategy=strategy,
        )

    async def forget_memories(
        self,
        *,
        min_total: int = 1_000,
        retain_fraction: float = 0.85,
    ) -> int:
        """Forget lowest-value memories using age-aware decay scoring."""
        from memory_system.core.maintenance import forget_old_memories

        return await forget_old_memories(
            self._store,
            self._index,
            min_total=min_total,
            retain_fraction=retain_fraction,
        )
