# enhanced_store.py — Enhanced memory store for Unified Memory System
"""Enhanced memory store with health checking and statistics."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, MutableMapping, Sequence, cast

import numpy as np
from cryptography.fernet import Fernet

from memory_system.config.settings import UnifiedSettings
from memory_system.core.faiss_vector_store import FaissVectorStore
from memory_system.core.interfaces import MetaStore, VectorStore
from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.summarization import SummaryStrategy
from memory_system.unified_memory import _get_ranking_weights

__all__ = ["EnhancedMemoryStore", "HealthComponent"]

log = logging.getLogger(__name__)


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
        db_cfg = settings.database
        self.meta_store: MetaStore = SQLiteMemoryStore(
            dsn,
            pool_size=db_cfg.connection_pool_size,
            wal=db_cfg.wal,
            synchronous=db_cfg.synchronous,
            page_size=db_cfg.page_size,
            cache_size=db_cfg.cache_size,
        )
        self.vector_store: VectorStore = FaissVectorStore(settings)
        self._index_lock = asyncio.Lock()
        self._ef_search = getattr(settings.model, "hnsw_ef_search", 64)

        # Helper for reinforcement/decay/score operations
        self._dynamics = MemoryDynamics(self.meta_store)

        # Backwards-compat for legacy tests that access private fields:
        self._store = self.meta_store  # type: ignore[attr-defined]
        self._index = getattr(self.vector_store, "index", None)  # type: ignore[attr-defined]

        vec_path = settings.database.vec_path

        try:
            self._memory_count = int(self.vector_store.stats().total_vectors)
        except Exception:
            self._memory_count = 0

        async def _save_index() -> None:
            # Persist the Faiss index after a DB commit
            await asyncio.to_thread(self.vector_store.save, str(vec_path))

        if hasattr(self.meta_store, "add_commit_hook"):
            self.meta_store.add_commit_hook(_save_index)  # type: ignore[attr-defined]

        self._closed = False

        # Optional: auto-tune HNSW search quality using control queries
        self._control_queries: list[tuple[np.ndarray, set[str]]] = []
        self._recall_target = 0.90
        self._monitor_interval = 60.0
        self._min_ef_search = max(16, self._ef_search // 2)
        self._max_ef_search = self._ef_search * 4
        self._monitor_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start background tasks for the store."""
        loop = asyncio.get_running_loop()
        self._monitor_task = loop.create_task(self._monitor_recall_loop())

    async def close(self) -> None:
        """Close the store."""
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(Exception):
                await self._monitor_task
        if hasattr(self.meta_store, "aclose"):
            await self.meta_store.aclose()  # type: ignore[attr-defined]
        self._closed = True

    async def get_health(self) -> HealthComponent:
        """Get health status."""
        uptime = int(time.time() - self._start_time)
        checks: dict[str, bool] = {}

        try:
            await self.meta_store.ping()
            checks["database"] = True
        except Exception:  # pragma: no cover
            checks["database"] = False

        try:
            _ = self.vector_store.stats().total_vectors
            checks["index"] = True
        except Exception:  # pragma: no cover
            checks["index"] = False

        checks.setdefault("embedding_service", True)

        healthy = all(checks.values())
        message = "All systems operational" if healthy else "Degraded"
        return HealthComponent(healthy=healthy, message=message, uptime=uptime, checks=checks)

    async def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if self._closed:
            raise RuntimeError("store is closed")
        return {
            "total_memories": self._memory_count,
            "index_size": getattr(self.vector_store.stats(), "total_vectors", 0),
            "cache_stats": {"hit_rate": 0.0},
            "buffer_size": 0,
            "uptime_seconds": int(time.time() - self._start_time),
        }

    # ----------------------------- admin/ops ---------------------------------

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
        except asyncio.CancelledError:  # pragma: no cover
            return

    async def _evaluate_recall(self) -> None:
        """Evaluate recall on control queries and adjust ef_search (HNSW only)."""
        if not self._control_queries:
            return
        if not hasattr(self.vector_store, "ef_search"):
            return

        recalls: list[float] = []
        for vec, expected in self._control_queries:
            k = max(len(expected), 10)
            ids, _ = await asyncio.to_thread(self.vector_store.search, vec, k=k)
            if expected:
                recalls.append(len(set(ids) & expected) / len(expected))
        if not recalls:
            return

        avg_recall = float(sum(recalls) / len(recalls))
        cur = int(getattr(self.vector_store, "ef_search", 64))
        if avg_recall < self._recall_target and cur < self._max_ef_search:
            new_ef = min(cur * 2, self._max_ef_search)
            # Apply by issuing a safe search call with the new ef_search
            await asyncio.to_thread(
                self.vector_store.search,
                self._control_queries[0][0],
                k=1,
                ef_search=new_ef,
            )
        elif avg_recall > self._recall_target + 0.05 and cur > self._min_ef_search:
            new_ef = max(cur // 2, self._min_ef_search)
            await asyncio.to_thread(
                self.vector_store.search,
                self._control_queries[0][0],
                k=1,
                ef_search=new_ef,
            )

    # ------------------------------------------------------------------
    # Public API used by routes/tests
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
        async with self._index_lock:
            await asyncio.to_thread(
                self.vector_store.add,
                [mem.id],
                np.asarray([embedding], dtype=np.float32),
                modality=modality,
            )
        # Save the index eagerly (a commit hook will also persist after DB commit)
        await asyncio.to_thread(self.vector_store.save, str(self.settings.database.vec_path))
        self._memory_count += 1
        return mem

    # ---- Memory dynamics ---------------------------------------------------

    async def reinforce(
        self,
        memory_id: str,
        amount: float = 0.1,
        *,
        valence_delta: float | None = None,
        intensity_delta: float | None = None,
    ) -> Memory:
        """Delegate reinforcement to :class:`MemoryDynamics`."""
        return await self._dynamics.reinforce(
            memory_id,
            amount,
            valence_delta=valence_delta,
            intensity_delta=intensity_delta,
        )

    def score(self, memory: Memory) -> float:
        """Return the time-decayed ranking score for ``memory``."""
        return self._dynamics.score(memory)

    @staticmethod
    def decay(
        *,
        importance: float,
        valence: float,
        emotional_intensity: float,
        age_days: float,
    ) -> float:
        """Expose :func:`MemoryDynamics.decay` for convenience."""
        return MemoryDynamics.decay(
            importance=importance,
            valence=valence,
            emotional_intensity=emotional_intensity,
            age_days=age_days,
        )

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
            async with self._index_lock:
                await asyncio.to_thread(
                    self.vector_store.add,
                    [mem.id],
                    np.asarray([vec], dtype=np.float32),
                    modality=modality,
                )

        await self.meta_store.add_many(mems)
        await asyncio.to_thread(self.vector_store.save, str(self.settings.database.vec_path))
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
            vec = np.asarray(item["embedding"], dtype=np.float32)
            batch.append((modality, mem, vec))

            if len(batch) >= batch_size:
                await self.meta_store.add_many([m for _, m, _ in batch])
                for mod, m, v in batch:
                    async with self._index_lock:
                        await asyncio.to_thread(
                            self.vector_store.add,
                            [m.id],
                            np.asarray([v], dtype=np.float32),
                            modality=mod,
                        )
                total += len(batch)
                batch.clear()

        if batch:
            await self.meta_store.add_many([m for _, m, _ in batch])
            for mod, m, v in batch:
                async with self._index_lock:
                    await asyncio.to_thread(
                        self.vector_store.add,
                        [m.id],
                        np.asarray([v], dtype=np.float32),
                        modality=mod,
                    )
            total += len(batch)

        await asyncio.to_thread(self.vector_store.save, str(self.settings.database.vec_path))
        self._memory_count += total
        return total

    async def semantic_search(
        self,
        *,
        vector: list[float],
        k: int = 5,
        return_distance: bool = False,
        ef_search: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
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
        modality:
            Modality name to route search to a corresponding sub-index.

        Returns
        -------
        list[Any]
            A list of memories (optionally paired with their distance).
        """
        # ANN search with over-sampling to improve recall
        vec = np.asarray(vector, dtype=np.float32)
        async with self._search_lock:
            ids, dists = await asyncio.to_thread(
                self.vector_store.search,
                vec,
                k=max(k * 5, k),
                modality=modality,
                ef_search=ef_search,
            )
        if not ids:
            return []

        md = dict(metadata_filter or {})
        md.setdefault("modality", modality)
        weights = _get_ranking_weights()
        allowed = await self.meta_store.top_n_by_score(
            k,
            level=level,
            metadata_filter=md,
            weights=weights,
            ids=list(ids),
        )
        if return_distance:
            dist_map = dict(zip(ids, dists, strict=True))
            return [(m, float(dist_map[m.id])) for m in allowed]
        return list(allowed)

    async def list_memories(self, user_id: str | None = None) -> list[Memory]:
        """List memories, optionally filtering by ``user_id``."""
        if user_id:
            return await self.meta_store.search(metadata_filters={"user_id": user_id})
        return await self.meta_store.search(limit=1000)

    # ---- Long-term maintenance ----------------------------------------------

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
            getattr(self.vector_store, "index", None),
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
            getattr(self.vector_store, "index", None),
            min_total=min_total,
            retain_fraction=retain_fraction,
        )
