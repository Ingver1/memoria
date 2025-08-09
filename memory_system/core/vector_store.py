"""memory_system.core.vector_store
=================================
Asynchronous FAISS‑based vector store with automatic background maintenance.

This module provides:
* ``AbstractVectorStore`` – minimal async interface, easy to mock.
* ``AsyncFaissHNSWStore`` – coroutine‑friendly implementation that wraps
  a FAISS HNSW index.  All heavy CPU‑bound FAISS calls run inside the
  default thread‑executor so the event loop never blocks.

Features
--------
* Built-in asynchronous reader/writer lock allows concurrent searches while
    writes remain exclusive.
* **JSON metadata** stored alongside IDs in a lightweight *sidecar*
  SQLite table – keeps FAISS fast and queries flexible.
* Background task (`_maintenance_loop`) performs **compaction** &
  **replication** every *maintenance_interval* seconds.
* Clean `await store.close()` shuts down the maintenance task and flushes
  the index to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import faiss

try:  # optional numpy
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from numpy.typing import NDArray
else:
    NDArray = Any


def _require_numpy() -> Any:
    if _np is None:
        raise ModuleNotFoundError("numpy is required for vector store operations. Install ai-memory[core].")
    return _np


from memory_system.utils.exceptions import StorageError, ValidationError
from memory_system.utils.rwlock import AsyncRWLock

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API / abstract interface
# ---------------------------------------------------------------------------
class AbstractVectorStore(ABC):
    """Interface that concrete stores must implement."""

    @abstractmethod
    async def add(self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]) -> list[str]:
        """Add vectors with associated metadata and return their IDs."""
        ...

    @abstractmethod
    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Search for nearest neighbours of ``vector``."""
        ...

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors identified by ``ids``."""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """Persist any in‑memory state to durable storage."""

    @abstractmethod
    async def close(self) -> None:
        """Flush + stop background tasks."""


# ---------------------------------------------------------------------------
# Implementation – AsyncFaissHNSWStore
# ---------------------------------------------------------------------------
class AsyncFaissHNSWStore(AbstractVectorStore):
    """Thread‑safe asynchronous wrapper around a FAISS HNSW index."""

    def __init__(
        self,
        dim: int,
        index_path: Path,
        maintenance_interval: int = 900,
    ) -> None:
        """Create a store backed by a FAISS index on disk."""
        self._dim = dim
        self._index_path = index_path
        self._rwlock = AsyncRWLock()
        self._maintenance_interval = maintenance_interval
        self._loop = asyncio.get_running_loop()

        # load or create index
        if index_path.exists():
            _LOGGER.info("Loading FAISS index from %s", index_path)
            self._index = faiss.read_index(str(index_path))
            # Load metadata if exists
            meta_path = index_path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    self._metadata = json.loads(meta_path.read_text())
                except Exception:
                    _LOGGER.warning("Failed to load metadata from %s", meta_path)
                    self._metadata = {}
            else:
                self._metadata = {}
        else:
            _LOGGER.info("Creating new FAISS HNSW index (dim=%d)", dim)
            base = faiss.IndexHNSWFlat(dim, 32)
            base.hnsw.efConstruction = 200
            self._index = faiss.IndexIDMap2(base)
            # write initial empty index so replica exists
            faiss.write_index(self._index, str(index_path))

        # metadata sidecar (id -> json str) stored in simple dict; caller may persist separately
        # _metadata is already set above, no need to redefine

        # start maintenance task
        self._stop_event = asyncio.Event()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------
    async def add(self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]) -> list[str]:
        """Store vectors with associated metadata and return new IDs."""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")

        ids = [str(uuid.uuid4()) for _ in vectors]
        async with self._rwlock.writer_lock():
            # FAISS needs contiguous array
            await self._loop.run_in_executor(
                None, self._index.add_with_ids, _to_faiss_array(vectors), _to_faiss_ids(ids)
            )
            for _id, meta in zip(ids, metadata, strict=False):
                self._metadata[_id] = meta
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Return ``k`` nearest vector IDs and distances for ``vector``."""
        async with self._rwlock.reader_lock():
            D, indices = self._index.search(_to_faiss_array([vector]), k)
        matches: list[tuple[str, float]] = []
        # Remove 'strict=False' for compatibility with Python <3.10
        for idx, dist in zip(indices[0], D[0], strict=False):
            if idx == -1:
                continue
            _id = _from_faiss_id(idx)
            matches.append((_id, float(dist)))
        return matches

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors by ``ids`` from the index and metadata store."""
        async with self._rwlock.writer_lock():
            id_array = _to_faiss_ids(ids)
            selector = faiss.IDSelectorBatch(id_array.size, faiss.swig_ptr(id_array))
            await self._loop.run_in_executor(None, self._index.remove_ids, selector)
            for _id in ids:
                self._metadata.pop(_id, None)

    async def flush(self) -> None:  # noqa: D401 (imperative)
        """Persist the FAISS index and metadata to disk."""
        async with self._rwlock.writer_lock():
            await self._loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_path))
            # simple metadata persistence
            (self._index_path.with_suffix(".meta.json")).write_text(json.dumps(self._metadata))

    async def close(self) -> None:
        """Stop background tasks and flush the index to disk."""
        self._stop_event.set()
        if self._maintenance_task:
            await self._maintenance_task
        await self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _maintenance_loop(self) -> None:
        """Periodically compacts & replicates the index on disk."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._maintenance_interval)
                if self._stop_event.is_set():
                    break
                await self.compact()
                await self.replicate()
            except Exception:  # pragma: no cover
                _LOGGER.exception("Vector store maintenance loop error")

    async def compact(self) -> None:
        """Writes the current index to disk, replacing previous blob."""
        _LOGGER.debug("Compacting FAISS index → %s", self._index_path)
        async with self._rwlock.writer_lock():
            await self._loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_path))

    async def replicate(self) -> None:
        """Makes a timestamped backup copy of the index blob."""
        ts = asyncio.get_running_loop().time()
        bak_path = self._index_path.with_suffix(f".{int(ts)}.bak")
        await self._loop.run_in_executor(None, shutil.copy2, self._index_path, bak_path)
        _LOGGER.debug("Replicated index to %s", bak_path)


# ---------------------------------------------------------------------------
# ––– Utility helpers –––
# ---------------------------------------------------------------------------
def _to_faiss_array(vectors: Sequence[Sequence[float]]) -> NDArray:
    """Convert a sequence of vectors to a 2-D float32 NumPy array."""
    np = _require_numpy()
    arr = cast(NDArray, np.array(vectors, dtype=np.float32))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _to_faiss_ids(ids: Sequence[str]) -> NDArray:
    """Map UUID strings to FAISS-compatible ``int64`` identifiers."""
    np = _require_numpy()
    return cast(
        NDArray,
        np.array([uuid.UUID(_id).int & ((1 << 64) - 1) for _id in ids], dtype=np.int64),
    )


def _from_faiss_id(idx: int) -> str:
    """Convert an ``int`` FAISS ID back to a UUID string."""
    return str(uuid.UUID(int=idx))


# ---------------------------------------------------------------------------
# Lightweight synchronous store used in the test-suite
# ---------------------------------------------------------------------------
import array as _array
import os
import sqlite3
import struct as _struct
import threading
import time
from typing import Sequence as _Seq

# numpy is optional; loaded lazily via _require_numpy


class VectorStore:
    """Very small local vector store used only for tests."""

    def __init__(self, base_path: Path, *, dim: int) -> None:
        """Initialise the store at ``base_path`` with vectors of dimension ``dim``."""
        self._base_path = Path(base_path)
        self._dim = dim
        self._bin_path = self._base_path.with_suffix(".bin")
        self._db_path = self._base_path.with_suffix(".db")

        self._bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._bin_path, "a+b")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db_lock = threading.Lock()
        self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, offset INTEGER)")
        self._conn.commit()

    # ------------------------------------------------------------------
    def _validate_vector(self, vector: _Seq[float] | np.ndarray) -> NDArray[np.float32]:
        """Validate ``vector`` and convert it to ``np.float32`` array."""
        np = _require_numpy()

        if isinstance(vector, np.ndarray):
            if vector.dtype != np.float32:
                raise ValidationError("vector dtype must be float32")
            if vector.ndim != 1:
                raise ValidationError("vector must be 1-D")
            arr: NDArray[np.float32] = vector.astype(np.float32, copy=False)
        else:
            arr = cast(NDArray[np.float32], np.asarray(vector, dtype=np.float32))
            if arr.ndim != 1:
                raise ValidationError("vector must be 1-D")

        if self._dim == 0:
            self._dim = arr.shape[0]
        if arr.shape[0] != self._dim:
            raise ValidationError(f"expected dim {self._dim}")

        return arr

    def add_vector(self, vector_id: str, vector: _Seq[float] | np.ndarray) -> None:
        """Add a single vector under ``vector_id``."""
        with self._db_lock:
            if self._conn.execute("SELECT 1 FROM vectors WHERE id=?", (vector_id,)).fetchone():
                raise ValidationError("duplicate id")
            arr = self._validate_vector(vector)
            self._file.seek(0, os.SEEK_END)
            offset = self._file.tell()
            buf = _array.array("f", [float(x) for x in arr])
            self._file.write(buf.tobytes())
            self._conn.execute(
                "INSERT INTO vectors (id, offset) VALUES (?, ?)",
                (vector_id, offset),
            )

    def get_vector(self, vector_id: str) -> NDArray[np.float32]:
        """Return the stored vector for ``vector_id``."""
        np = _require_numpy()
        with self._db_lock:
            row = self._conn.execute(
                "SELECT offset FROM vectors WHERE id=?",
                (vector_id,),
            ).fetchone()
            if row is None:
                raise StorageError("Vector not found")
            offset = row[0]
            self._file.seek(offset)
            buf = self._file.read(self._dim * 4)
            arr: _array.array[float] = _array.array("f")
            arr.frombytes(buf)
            vec = cast(NDArray[np.float32], np.asarray(arr, dtype=np.float32))
            return vec

    def remove_vector(self, vector_id: str) -> None:
        """Delete the vector associated with ``vector_id``."""
        with self._db_lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE id=?", (vector_id,))
            if cur.rowcount == 0:
                raise StorageError("Vector not found")

    def list_ids(self) -> list[str]:
        """Return all stored vector IDs."""
        with self._db_lock:
            cur = self._conn.execute("SELECT id FROM vectors")
            try:
                ids: list[str] = []
                while True:
                    batch = cur.fetchmany(1000)
                    if not batch:
                        break
                    ids.extend(row[0] for row in batch)
                return ids
            finally:
                cur.close()

    async def flush(self) -> None:
        """Persist vectors and metadata to disk."""
        with self._db_lock:
            self._conn.commit()
            self._file.flush()

    async def async_flush(self) -> None:  # compatibility helper
        """Async wrapper around :meth:`flush` for API parity."""
        await self.flush()

    async def replicate(self) -> None:
        """Create a timestamped backup of the data file."""
        await self.flush()
        ts = int(time.time())
        bak_path = self._bin_path.with_suffix(f".{ts}.bak")
        shutil.copy2(self._bin_path, bak_path)

    def close(self) -> None:
        """Flush and close underlying file and database handles."""
        with self._db_lock:
            self._conn.commit()
            self._conn.close()
            self._file.close()


# Backwards compatibility alias for asynchronous FAISS store
VectorStoreAsync = AsyncFaissHNSWStore


def create_vector_store(
    *,
    backend: Literal["faiss", "qdrant"],
    dim: int,
    index_path: Path | None = None,
    **kwargs: Any,
) -> AbstractVectorStore:
    """Factory returning a vector store implementation based on ``backend``."""
    if backend == "faiss":
        if index_path is None:
            raise ValueError("index_path is required for FAISS backend")
        return AsyncFaissHNSWStore(dim=dim, index_path=index_path, **kwargs)
    if backend == "qdrant":
        from .qdrant_store import QdrantVectorStore

        return QdrantVectorStore(dim=dim, **kwargs)
    raise ValueError(f"Unsupported vector store backend: {backend}")


__all__ = [
    "AbstractVectorStore",
    "AsyncFaissHNSWStore",
    "VectorStore",
    "VectorStoreAsync",
    "create_vector_store",
]
