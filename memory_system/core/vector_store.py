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
* Single **`asyncio.Lock`** guards mutating ops (no `threading.RLock`).
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
from typing import Any

import faiss
import numpy as _np
from memory_system.utils.exceptions import StorageError, ValidationError

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API / abstract interface
# ---------------------------------------------------------------------------
class AbstractVectorStore(ABC):
    """Interface that concrete stores must implement."""

    @abstractmethod
    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]: ...

    @abstractmethod
    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]: ...

    @abstractmethod
    async def delete(self, ids: Sequence[str]) -> None: ...

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
        self._dim = dim
        self._index_path = index_path
        self._lock = asyncio.Lock()
        self._maintenance_interval = maintenance_interval
        self._loop = asyncio.get_running_loop()

        # load or create index
        if index_path.exists():
            _LOGGER.info("Loading FAISS index from %s", index_path)
            self._index = faiss.read_index(str(index_path))
        else:
            _LOGGER.info("Creating new FAISS HNSW index (dim=%d)", dim)
            base = faiss.IndexHNSWFlat(dim, 32)
            base.hnsw.efConstruction = 200
            self._index = faiss.IndexIDMap2(base)
            # write initial empty index so replica exists
            faiss.write_index(self._index, str(index_path))

        # metadata sidecar (id -> json str) stored in simple dict; caller may persist separately
        self._metadata: dict[str, dict[str, Any]] = {}

        # start maintenance task
        self._stop_event = asyncio.Event()
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------
    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")

        ids = [str(uuid.uuid4()) for _ in vectors]
        async with self._lock:
            # FAISS needs contiguous array
            await self._loop.run_in_executor(
                None, self._index.add_with_ids, _to_faiss_array(vectors), _to_faiss_ids(ids)
            )
            for _id, meta in zip(ids, metadata, strict=False):
                self._metadata[_id] = meta
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        async with self._lock:
            D, indices = await self._loop.run_in_executor(
                None, self._index.search, _to_faiss_array([vector]), k
            )
        matches: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], D[0], strict=False):
            if idx == -1:
                continue
            _id = _from_faiss_id(idx)
            matches.append((_id, float(dist)))
        return matches

    async def delete(self, ids: Sequence[str]) -> None:
        async with self._lock:
            id_array = _to_faiss_ids(ids)
            selector = faiss.IDSelectorBatch(id_array.size, faiss.swig_ptr(id_array))
            await self._loop.run_in_executor(None, self._index.remove_ids, selector)
            for _id in ids:
                self._metadata.pop(_id, None)

    async def flush(self) -> None:  # noqa: D401 (imperative)
        async with self._lock:
            await self._loop.run_in_executor(
                None, faiss.write_index, self._index, str(self._index_path)
            )
            # simple metadata persistence
            (self._index_path.with_suffix(".meta.json")).write_text(json.dumps(self._metadata))

    async def close(self) -> None:
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
                await self.compact()
                await self.replicate()
            except Exception:  # pragma: no cover
                _LOGGER.exception("Vector store maintenance loop error")

    async def compact(self) -> None:
        """Writes the current index to disk, replacing previous blob."""
        _LOGGER.debug("Compacting FAISS index → %s", self._index_path)
        async with self._lock:
            await self._loop.run_in_executor(
                None, faiss.write_index, self._index, str(self._index_path)
            )

    async def replicate(self) -> None:
        """Makes a timestamped backup copy of the index blob."""
        ts = asyncio.get_running_loop().time()
        bak_path = self._index_path.with_suffix(f".{int(ts)}.bak")
        await self._loop.run_in_executor(None, shutil.copy2, self._index_path, bak_path)
        _LOGGER.debug("Replicated index to %s", bak_path)


# ---------------------------------------------------------------------------
# ––– Utility helpers –––
# ---------------------------------------------------------------------------
def _to_faiss_array(vectors: Sequence[Sequence[float]]) -> _np.ndarray:
    arr = _np.array(vectors, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _to_faiss_ids(ids: Sequence[str]) -> _np.ndarray:
    int_ids = _np.array([int(uuid.UUID(_id)) % (2**63) for _id in ids], dtype="int64")
    return int_ids

def _from_faiss_id(idx: int) -> str:
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

import numpy as np


class VectorStore:
    """Very small local vector store used only for tests."""

    def __init__(self, base_path: Path, *, dim: int) -> None:
        self._base_path = Path(base_path)
        self._dim = dim
        self._bin_path = self._base_path.with_suffix(".bin")
        self._db_path = self._base_path.with_suffix(".db")

        self._bin_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._bin_path, "a+b")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db_lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, offset INTEGER)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    def _validate_vector(self, vector: _Seq[float] | np.ndarray) -> np.ndarray:
        def _f32(val: float) -> float:
            return float(_struct.unpack("f", _struct.pack("f", float(val)))[0])

        if isinstance(vector, np.ndarray):
            if getattr(vector, "dtype", np.float32) is not np.float32:
                raise ValidationError("vector dtype must be float32")
            if vector.ndim != 1:
                raise ValidationError("vector must be 1-D")
            arr_list = [_f32(x) for x in vector]
        else:
            for x in vector:
                if isinstance(x, (list, tuple, np.ndarray)):
                    raise ValidationError("vector must be 1-D")
            arr_list = [_f32(x) for x in vector]
        if self._dim == 0:
            self._dim = len(arr_list)
        if len(arr_list) != self._dim:
            raise ValidationError(f"expected dim {self._dim}")
        return np.asarray(arr_list, dtype=np.float32)

    def add_vector(self, vector_id: str, vector: _Seq[float] | np.ndarray) -> None:
        with self._db_lock:
            if self._conn.execute("SELECT 1 FROM vectors WHERE id=?", (vector_id,)).fetchone():
                raise ValidationError("duplicate id")
            arr = self._validate_vector(vector)
            self._file.seek(0, os.SEEK_END)
            offset = self._file.tell()
            buf = _array.array("f", [float(x) for x in arr])
            self._file.write(buf.tobytes())
            self._conn.execute("INSERT INTO vectors (id, offset) VALUES (?, ?)", (vector_id, offset))
            self._conn.commit()

    def get_vector(self, vector_id: str) -> np.ndarray:
        """Return the stored vector for ``vector_id``."""
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
            arr = _array.array("f")
            arr.frombytes(buf)
            return np.asarray(arr, dtype=np.float32)

    def remove_vector(self, vector_id: str) -> None:
        with self._db_lock:
            cur = self._conn.execute("DELETE FROM vectors WHERE id=?", (vector_id,))
            if cur.rowcount == 0:
                raise StorageError("Vector not found")
            self._conn.commit()

    def list_ids(self) -> list[str]:
        with self._db_lock:
            rows = self._conn.execute("SELECT id FROM vectors").fetchall()
            return [r[0] for r in rows]

    async def flush(self) -> None:
        with self._db_lock:
            self._conn.commit()
            self._file.flush()

    async def async_flush(self) -> None:  # compatibility helper
        await self.flush()

    async def replicate(self) -> None:
        await self.flush()
        ts = int(time.time())
        bak_path = self._bin_path.with_suffix(f".{ts}.bak")
        shutil.copy2(self._bin_path, bak_path)

    def close(self) -> None:
        with self._db_lock:
            self._conn.commit()
            self._conn.close()
            self._file.close()


# Backwards compatibility alias for asynchronous FAISS store
VectorStoreAsync = AsyncFaissHNSWStore

__all__ = [
    "AbstractVectorStore",
    "AsyncFaissHNSWStore",
    "VectorStore",
]
