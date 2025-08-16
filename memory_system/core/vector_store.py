"""
memory_system.core.vector_store.

=================================
Asynchronous FAISS-based vector store with automatic background maintenance.

This module provides:
* ``AbstractVectorStore`` - minimal async interface, easy to mock.
* ``AsyncFaissHNSWStore`` - coroutine-friendly implementation that wraps
  a FAISS HNSW index.  All heavy CPU-bound FAISS calls run inside the
  default thread-executor so the event loop never blocks.

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
* The lightweight synchronous ``VectorStore`` can be used as a context
  manager: ``with VectorStore(path, dim) as store: ...`` to ensure files
  and database handles are closed automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from memory_system.utils.dependencies import require_faiss, require_numpy
from memory_system.utils.loop import get_loop

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import faiss
else:
    faiss = require_faiss()

Float32Array: TypeAlias = NDArray[np.float32]
Int64Array: TypeAlias = NDArray[np.int64]


from memory_system.utils.exceptions import StorageError, ValidationError
from memory_system.utils.rwlock import AsyncRWLock
from memory_system.utils.security import EncryptionManager

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from cryptography.fernet import Fernet as _Fernet

Fernet: type[_Fernet] | None
try:
    from cryptography.fernet import Fernet as _FernetCls
except ImportError:  # pragma: no cover - optional dependency
    Fernet = None
else:
    Fernet = _FernetCls

ENVELOPE_MANAGER = EncryptionManager()

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vector store protocol and registry
# ---------------------------------------------------------------------------
VectorStoreT = TypeVar("VectorStoreT", bound="VectorStoreProtocol")


def register_vector_store(name: str) -> Callable[[type[VectorStoreT]], type[VectorStoreT]]:
    """Class decorator used to register vector store implementations."""

    def decorator(cls: type[VectorStoreT]) -> type[VectorStoreT]:
        VECTOR_STORE_REGISTRY[name] = cls
        return cls

    return decorator


class VectorStoreProtocol(Protocol):
    """Structural protocol that all vector stores must implement."""

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        """Add vectors with associated metadata and return their IDs."""

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """Search for nearest neighbours of ``vector``."""

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove vectors identified by ``ids``."""

    async def flush(self) -> None:
        """Persist any in-memory state to durable storage."""

    async def close(self) -> None:
        """Flush and stop background tasks."""


# Backwards compatibility alias
AbstractVectorStore = VectorStoreProtocol


# Global registry storing backend name -> implementation mapping
VECTOR_STORE_REGISTRY: dict[str, type[VectorStoreProtocol]] = {}


# ---------------------------------------------------------------------------
# Implementation – AsyncFaissHNSWStore
# ---------------------------------------------------------------------------
@register_vector_store("faiss")
class AsyncFaissHNSWStore:
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
        self._loop = get_loop()

        # load or create index
        if index_path.exists():
            _LOGGER.info("Loading FAISS index from %s", index_path)
            self._index = faiss.read_index(str(index_path))
            # Load metadata if exists
            meta_path = index_path.with_suffix(".meta.json")
            if meta_path.exists():
                try:
                    self._metadata = json.loads(meta_path.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
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
    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
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

    async def flush(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        async with self._rwlock.writer_lock():
            await self._loop.run_in_executor(
                None, faiss.write_index, self._index, str(self._index_path)
            )
            # simple metadata persistence
            (self._index_path.with_suffix(".meta.json")).write_text(json.dumps(self._metadata))

    async def close(self) -> None:
        """Stop background tasks and flush the index to disk."""
        self._stop_event.set()
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await self._maintenance_task
        await self.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _maintenance_loop(self) -> None:
        """Periodically compacts & replicates the index on disk."""
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._maintenance_interval
                    )
                except TimeoutError:
                    try:
                        await self.compact()
                        await self.replicate()
                    except (OSError, RuntimeError):  # pragma: no cover
                        _LOGGER.exception("Vector store maintenance loop error")
                else:
                    break
        except asyncio.CancelledError:
            pass

    async def compact(self) -> None:
        """Writes the current index to disk, replacing previous blob."""
        _LOGGER.debug("Compacting FAISS index → %s", self._index_path)
        async with self._rwlock.writer_lock():
            await self._loop.run_in_executor(
                None, faiss.write_index, self._index, str(self._index_path)
            )

    async def replicate(self) -> None:
        """Makes a timestamped backup copy of the index blob."""
        ts = int(time.time())
        bak_path = self._index_path.with_suffix(f".{ts}.bak")
        await self._loop.run_in_executor(None, shutil.copy2, self._index_path, bak_path)
        _LOGGER.debug("Replicated index to %s", bak_path)


# ---------------------------------------------------------------------------
# ––– Utility helpers –––
# ---------------------------------------------------------------------------
def _to_faiss_array(vectors: Sequence[Sequence[float]]) -> Float32Array:
    """Convert a sequence of vectors to a 2-D float32 NumPy array."""
    np = require_numpy()
    arr = cast("Float32Array", np.array(vectors, dtype=np.float32))
    if arr.ndim == 1:
        arr = cast("Float32Array", arr.reshape(1, -1))
    return arr


def _to_faiss_ids(ids: Sequence[str]) -> Int64Array:
    """Map UUID strings to FAISS-compatible ``int64`` identifiers."""
    np = require_numpy()
    return cast(
        "Int64Array",
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
import threading
from collections.abc import Sequence as _Seq

# numpy is optional; loaded lazily via ``require_numpy``


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
        # Enable WAL mode for better concurrent write behaviour and reduced
        # writer blocking.  ``NORMAL`` synchronous level avoids an fsync on
        # every transaction while still providing crash safety when a
        # checkpoint is run.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._db_lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (id TEXT PRIMARY KEY, offset INTEGER, ciphertext BLOB, nonce BLOB, cek_wrapped BLOB, kek_id TEXT)"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    def _validate_vector(self, vector: _Seq[float] | ndarray) -> Float32Array:
        """Validate ``vector`` and convert it to ``np.float32`` array."""
        np = require_numpy()
        using_stub = bool(getattr(np, "__stub__", False))

        if isinstance(vector, np.ndarray):
            dtype = getattr(vector, "dtype", np.float32)
            if dtype != np.float32:
                raise ValidationError("vector dtype must be float32")
            if getattr(vector, "ndim", 1) != 1:
                raise ValidationError("vector must be 1-D")
            if using_stub:
                arr = cast("Float32Array", np.asarray(vector, dtype=float))
            else:
                arr = cast("Float32Array", vector.astype(np.float32))
        else:
            arr = cast(
                "Float32Array",
                np.asarray(vector, dtype=(float if using_stub else np.float32)),
            )
            if arr.ndim != 1:
                raise ValidationError("vector must be 1-D")

        if self._dim == 0:
            self._dim = arr.shape[0]
        if arr.shape[0] != self._dim:
            raise ValidationError(f"expected dim {self._dim}")

        return arr

    def add_vector(self, vector_id: str, vector: _Seq[float] | ndarray) -> None:
        """Add a single vector under ``vector_id``."""
        np = require_numpy()
        using_stub = bool(getattr(np, "__stub__", False))
        with self._db_lock:
            if self._conn.execute("SELECT 1 FROM vectors WHERE id=?", (vector_id,)).fetchone():
                raise ValidationError("duplicate id")
            arr = self._validate_vector(vector)
            self._file.seek(0, os.SEEK_END)
            offset = self._file.tell()
            code = "f" if not using_stub else "d"
            buf = _array.array(code, [float(x) for x in arr])
            raw = buf.tobytes()
            self._file.write(raw)
            if Fernet is not None:
                cek = Fernet.generate_key()
                cipher = Fernet(cek).encrypt(raw)
                wrapped = ENVELOPE_MANAGER.encrypt(cek.decode())
                self._conn.execute(
                    "INSERT INTO vectors (id, offset, ciphertext, nonce, cek_wrapped, kek_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (vector_id, offset, cipher, b"", wrapped, "local"),
                )
            else:
                self._conn.execute(
                    "INSERT INTO vectors (id, offset) VALUES (?, ?)",
                    (vector_id, offset),
                )

    def get_vector(self, vector_id: str) -> NDArray[np.float32]:
        """Return the stored vector for ``vector_id``."""
        np = require_numpy()
        using_stub = bool(getattr(np, "__stub__", False))
        with self._db_lock:
            row = self._conn.execute(
                "SELECT offset FROM vectors WHERE id=?",
                (vector_id,),
            ).fetchone()
            if row is None:
                raise StorageError("Vector not found")
            offset = row[0]
            self._file.seek(offset)
            size = 4 if not using_stub else 8
            code = "f" if not using_stub else "d"
            buf = self._file.read(self._dim * size)
            arr: _array.array[float] = _array.array(code)
            arr.frombytes(buf)
            vec = cast(
                "Float32Array",
                np.asarray(arr, dtype=(np.float32 if not using_stub else float)),
            )
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
            # Merge the WAL into the main database to keep file sizes bounded
            # across flushes.  ``TRUNCATE`` mode resets the WAL file entirely.
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._file.flush()

    async def async_flush(self) -> None:  # compatibility helper
        """Async wrapper around :meth:`flush` for API parity."""
        await self.flush()

    async def replicate(self) -> None:
        """Create timestamped backups of vector and metadata files."""
        await self.flush()
        ts = int(time.time())
        bin_bak = self._bin_path.with_name(f"{self._bin_path.name}.{ts}.bak")
        db_bak = self._db_path.with_name(f"{self._db_path.name}.{ts}.bak")
        shutil.copy2(self._bin_path, bin_bak)
        shutil.copy2(self._db_path, db_bak)

    def wipe(self) -> None:
        """Deterministically remove all stored vectors and metadata."""
        with self._db_lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()
            self._file.seek(0)
            self._file.truncate()

    @classmethod
    def restore_latest_backup(cls, base_path: Path) -> None:
        """Restore bin and DB files from the most recent ``.bak`` snapshots."""
        parent = base_path.parent
        stem = base_path.name
        bin_baks = sorted(parent.glob(f"{stem}.bin.*.bak"))
        db_baks = sorted(parent.glob(f"{stem}.db.*.bak"))
        if not bin_baks or not db_baks:
            raise FileNotFoundError("no backup available")
        shutil.copy2(bin_baks[-1], parent / f"{stem}.bin")
        shutil.copy2(db_baks[-1], parent / f"{stem}.db")

    def close(self) -> None:
        """Flush and close underlying file and database handles."""
        with self._db_lock:
            self._conn.commit()
            # Ensure WAL contents are merged before closing to avoid leaving
            # a large -wal file on disk.
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._file.close()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> VectorStore:
        """Enter the runtime context and return ``self``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and close the store."""
        self.close()


# Backwards compatibility alias for asynchronous FAISS store
VectorStoreAsync = AsyncFaissHNSWStore


def create_vector_store(
    *,
    backend: str,
    dim: int,
    index_path: Path | None = None,
    **kwargs: Any,
) -> VectorStoreProtocol:
    """
    Return a vector store implementation based on ``backend``.

    ``backend`` may either be one of the built-in names or a fully qualified
    ``module:Class`` path pointing at a custom implementation. Registered
    backends are looked up in :data:`VECTOR_STORE_REGISTRY` first.
    """
    if backend == "faiss" and index_path is None:
        raise ValueError("index_path is required for FAISS backend")

    params: dict[str, Any] = {"dim": dim, **kwargs}
    if index_path is not None:
        params["index_path"] = index_path

    if backend in VECTOR_STORE_REGISTRY:
        cls = VECTOR_STORE_REGISTRY[backend]
        return cls(**params)

    import importlib

    # Allow explicit module path via "module:Class" or "module.Class"
    if ":" in backend or "." in backend:
        module_name, _, class_name = backend.replace(":", ".").rpartition(".")
    else:
        module_name = f"memory_system.core.{backend}_store"
        class_name = f"{backend.capitalize()}VectorStore"

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    instance = cast("VectorStoreProtocol", cls(**params))
    VECTOR_STORE_REGISTRY.setdefault(backend, cls)
    return instance


class VectorStoreFactory:
    """Factory that manages vector store backends at runtime."""

    def __init__(
        self,
        *,
        backend: str,
        dim: int,
        index_path: Path | None = None,
        **config: Any,
    ) -> None:
        self._backend = backend
        self._dim = dim
        self._index_path = index_path
        self._config = config
        self._instance: VectorStoreProtocol | None = None

    def get(self) -> VectorStoreProtocol:
        """Return the current backend instance, creating it if necessary."""
        if self._instance is None:
            self._instance = create_vector_store(
                backend=self._backend,
                dim=self._dim,
                index_path=self._index_path,
                **self._config,
            )
        return self._instance

    async def swap(self, backend: str, **config: Any) -> VectorStoreProtocol:
        """Swap to a different backend at runtime."""
        if self._instance is not None:
            await self._instance.close()
        self._backend = backend
        self._config = config
        self._instance = create_vector_store(backend=backend, dim=self._dim, **config)
        return self._instance


__all__ = [
    "VECTOR_STORE_REGISTRY",
    "AbstractVectorStore",
    "AsyncFaissHNSWStore",
    "VectorStore",
    "VectorStoreAsync",
    "VectorStoreFactory",
    "VectorStoreProtocol",
    "create_vector_store",
    "register_vector_store",
]
