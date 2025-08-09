# enhanced_store.py — Enhanced memory store for Unified Memory System
#
# Version: v{__version__}
"""Enhanced memory store with health checking and statistics."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Sequence, cast

import numpy as np
from cryptography.fernet import Fernet

__all__ = ["EnhancedMemoryStore", "HealthComponent"]

log = logging.getLogger(__name__)

from memory_system.config.settings import UnifiedSettings
from memory_system.core.faiss_vector_store import FaissVectorStore
from memory_system.core.interfaces import MetaStore, VectorStore
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

        dsn = settings.get_database_url()
        self.meta_store: MetaStore = SQLiteMemoryStore(dsn)
        self.vector_store: VectorStore = FaissVectorStore(settings)
        # Backwards compatibility for tests accessing internal attributes
        self._store = self.meta_store  # type: ignore[assignment]
        self._index = self.vector_store.index  # type: ignore[assignment]

        vec_path = settings.database.vec_path

        self._memory_count = self.vector_store.stats().total_vectors

        async def _save_index() -> None:
            await asyncio.to_thread(self.vector_store.save, str(vec_path))

        self.meta_store.add_commit_hook(_save_index)

        self._closed = False

        self._control_queries: list[tuple[np.ndarray, set[str]]] = []
        self._recall_target = 0.9
        self._monitor_interval = 60.0
        self._min_ef_search = max(16, settings.model.hnsw_ef_search // 2)
        self._max_ef_search = settings.model.hnsw_ef_search * 4
        self._monitor_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background tasks for the store."""
        loop = asyncio.get_running_loop()
        self._monitor_task = loop.create_task(self._monitor_recall_loop())

    async def get_health(self) -> HealthComponent:
        """Get health status."""
        uptime = int(time.time() - self._start_time)
        checks: dict[str, bool] = {}

        try:
            await self.meta_store.ping()
            checks["database"] = True
        except Exception:  # pragma: no cover - connection issues
            checks["database"] = False

        try:
            _ = self.vector_store.stats().total_vectors
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
            "index_size": self.vector_store.stats().total_vectors,
            "cache_stats": {"hit_rate": 0.0},
            "buffer_size": 0,
            "uptime_seconds": int(time.time() - self._start_time),
        }

    async def close(self) -> None:
        """Close the store."""
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(Exception):
                await self._monitor_task
        await self.meta_store.aclose()
        self._closed = True

    def add_control_query(self, embedding: list[float], expected_ids: list[str]) -> None:
        """Register a control query used for recall monitoring."""
        vec = np.asarray(embedding, dtype=np.float32)
        self._control_queries.append((vec, set(expected_ids)))

    async def _monitor_recall_loop(self) -> None:
        """Background task that monitors index recall and tunes ef_search."""
        try:
            while True:
                await asyncio.sleep(self._monitor_interval)
                await self._evaluate_recall()
        except asyncio.CancelledError:  # pragma: no cover - task cancelled on shutdown
            return

    async def _evaluate_recall(self) -> None:
        """Evaluate recall on control queries and adjust ef_search."""
        if not self._control_queries:
            return
        recalls: list[float] = []
        for vec, expected in self._control_queries:
            k = max(len(expected), 10)
            ids, _ = self.vector_store.search(vec, k=k)
            if expected:
                recalls.append(len(set(ids) & expected) / len(expected))
        if not recalls:
            return
        avg_recall = float(sum(recalls) / len(recalls))
        if avg_recall < self._recall_target and self.vector_store.ef_search < self._max_ef_search:
            new_ef = min(self.vector_store.ef_search * 2, self._max_ef_search)
            self.vector_store.search(self._control_queries[0][0], k=1, ef_search=new_ef)
        elif avg_recall > self._recall_target + 0.05 and self.vector_store.ef_search > self._min_ef_search:
            new_ef = max(self.vector_store.ef_search // 2, self._min_ef_search)
            self.vector_store.search(self._control_queries[0][0], k=1, ef_search=new_ef)

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
        modality: str = "text",
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
            modality=modality,
        )
        await self.meta_store.add(mem)
        self.vector_store.add([mem.id], np.asarray([embedding], dtype=np.float32), modality=modality)
        self.vector_store.save(str(self.settings.database.vec_path))
        self._memory_count += 1
        return mem

    async def add_memories_batch(self, items: Sequence[dict[str, Any]]) -> list[Memory]:
        """Add multiple memories with embeddings in one batch."""

        mems: list[Memory] = []
        for item in items:
            text = item["text"]
            text_to_store = text
            if self.settings.security.encrypt_at_rest:
                f = Fernet(self.settings.security.encryption_key.encode())
                text_to_store = f.encrypt(text.encode()).decode()
            modality = item.get("modality", "text")
            mem = Memory(
                id=str(uuid.uuid4()),
                text=text_to_store,
                created_at=dt.datetime.fromtimestamp(item.get("created_at", time.time()), tz=dt.timezone.utc),
                importance=item.get("importance", 0.0),
                valence=item.get("valence", 0.0),
                emotional_intensity=item.get("emotional_intensity", 0.0),
                metadata={"role": item.get("role"), "tags": item.get("tags", [])},
                modality=modality,
            )
            mems.append(mem)
            vec = np.asarray(item["embedding"], dtype=np.float32)
            self.vector_store.add([mem.id], np.asarray([vec], dtype=np.float32), modality=modality)
        await self.meta_store.add_many(mems)
        self.vector_store.save(str(self.settings.database.vec_path))
        self._memory_count += len(mems)
        return mems

    async def add_memories_streaming(
        self,
        iterator: Iterable[dict[str, Any]] | AsyncIterator[dict[str, Any]],
        *,
        batch_size: int = 100,
    ) -> int:
        """Stream memories into the store and index without full buffering."""

        async def _aiter(it: Iterable[dict[str, Any]] | AsyncIterator[dict[str, Any]]):
            if hasattr(it, "__aiter__"):
                async for item in cast(AsyncIterator[dict[str, Any]], it):
                    yield item
            else:
                for item in cast(Iterable[dict[str, Any]], it):
                    yield item

        batch: list[tuple[str, Memory, np.ndarray]] = []
        total = 0
        async for item in _aiter(iterator):
            text = item["text"]
            text_to_store = text
            if self.settings.security.encrypt_at_rest:
                f = Fernet(self.settings.security.encryption_key.encode())
                text_to_store = f.encrypt(text.encode()).decode()
            modality = item.get("modality", "text")
            mem = Memory(
                id=str(uuid.uuid4()),
                text=text_to_store,
                created_at=dt.datetime.fromtimestamp(item.get("created_at", time.time()), tz=dt.timezone.utc),
                importance=item.get("importance", 0.0),
                valence=item.get("valence", 0.0),
                emotional_intensity=item.get("emotional_intensity", 0.0),
                metadata={"role": item.get("role"), "tags": item.get("tags", [])},
                modality=modality,
            )
            batch.append((modality, mem, np.asarray(item["embedding"], dtype=np.float32)))
            if len(batch) >= batch_size:
                await self.meta_store.add_many([m for _, m, _ in batch])
                for mod, m, vec in batch:
                    self.vector_store.add([m.id], np.asarray([vec], dtype=np.float32), modality=mod)
                total += len(batch)
                batch.clear()
        if batch:
            await self.meta_store.add_many([m for _, m, _ in batch])
            for mod, m, vec in batch:
                self.vector_store.add([m.id], np.asarray([vec], dtype=np.float32), modality=mod)
            total += len(batch)
        self.vector_store.save(str(self.settings.database.vec_path))
        self._memory_count += total
        return total

    async def semantic_search(
        self,
        *,
        vector: list[float],
        k: int = 5,
        return_distance: bool = False,
        ef_search: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        level: int | None = None,
        modality: str = "text",
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
        # Always filter by modality (and optional user metadata), retrieving
        # matching rows in a single query so that we can avoid per-ID lookups.
        metadata_filter = dict(metadata_filter or {})
        metadata_filter.setdefault("modality", modality)
        total = self.vector_store.stats(modality).total_vectors or k
        search_k = min(k * 5, max(1, total))

        vec = np.asarray(vector, dtype=np.float32)

        ids, dists = self.vector_store.search(vec, k=search_k, modality=modality, ef_search=ef_search)
        candidates = list(zip(ids, dists, strict=False))

        allowed_mems = await self.meta_store.search(
            metadata_filters=metadata_filter,
            limit=total,  # cover whole corpus; store caps internally if needed
            level=level,
        )
        allowed_map = {m.id: m for m in allowed_mems}
        if not allowed_map:
            return []
        candidates = [(_id, dist) for _id, dist in candidates if _id in allowed_map][:k]

        # Materialize Memory objects from the cached mapping and build result
        if return_distance:
            return [(allowed_map[_id], float(dist)) for _id, dist in candidates]
        return [allowed_map[_id] for _id, _ in candidates]

    async def list_memories(self, user_id: str | None = None) -> list[Memory]:
        """List memories, optionally filtering by ``user_id``."""
        if user_id:
            return await self.meta_store.search(metadata_filters={"user_id": user_id})
        return await self.meta_store.search(limit=1000)

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
            self.meta_store,
            self.vector_store.index,
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
            self.meta_store,
            self.vector_store.index,
            min_total=min_total,
            retain_fraction=retain_fraction,
        )
