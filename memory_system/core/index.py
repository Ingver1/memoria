# index.py — FAISS‑based ANN index for Unified Memory System
#
# Version: 0.8‑alpha

"""Vector‑similarity index built on top of **FAISS** (*IndexHNSWFlat*) with
ID mapping, basic statistics, dynamic search tuning and Prometheus hooks.

```python
from memory_system.core.index import FaissHNSWIndex
idx = FaissHNSWIndex(dim=768)
idx.add_vectors(["id‑1", "id‑2"], np.random.rand(2, 768))
ids, dist = idx.search(np.random.rand(768), k=5)
```

The class is thread‑safe because all state mutations are protected by a :class:`threading.RLock`.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from time import perf_counter

import faiss
import numpy as np
from memory_system.utils.exceptions import StorageError
from memory_system.utils.metrics import prometheus_counter
from numpy import ndarray as NDArray

log = logging.getLogger(__name__)

# ───────────────────── Prometheus collectors ─────────────────────

_VEC_ADDED = prometheus_counter("ums_vectors_added_total", "Vectors added to ANN index")
_VEC_DELETED = prometheus_counter("ums_vectors_deleted_total", "Vectors deleted from ANN index")
_QUERY_CNT = prometheus_counter("ums_ann_queries_total", "ANN queries executed")
_QUERY_ERR = prometheus_counter("ums_ann_query_errors_total", "Errors while querying ANN index")

# ────────────────────────── Exceptions ───────────────────────────


class ANNIndexError(StorageError, ValueError):
    """Raised for duplicate IDs, dimension mismatch, or internal FAISS errors."""


# ─────────────────────────── Dataclass ────────────────────────────


@dataclass(slots=True)
class IndexStats:
    dim: int
    total_vectors: int = 0
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    last_rebuild: float | None = None
    extra: dict[str, int | float] = field(default_factory=dict)


# ────────────────────────── Main class ────────────────────────────


class FaissHNSWIndex:
    """High‑level wrapper over *faiss.IndexHNSWFlat* with ID mapping and stats."""

    DEFAULT_EF_CONSTRUCTION = int(os.getenv("UMS_EF_CONSTRUCTION", "128"))
    DEFAULT_HNSW_M = int(os.getenv("UMS_HNSW_M", "32"))

    def __init__(
        self,
        dim: int,
        *,
        ef_construction: int | None = None,
        M: int | None = None,
        space: str = "cosine",
    ) -> None:
        self.dim = dim
        self.space = space
        self._lock: RLock = RLock()

        # Build underlying FAISS index -----------------------------------
        metric = faiss.METRIC_INNER_PRODUCT if space == "cosine" else faiss.METRIC_L2
        base = faiss.IndexHNSWFlat(dim, M or self.DEFAULT_HNSW_M, metric)
        base.hnsw.efConstruction = ef_construction or self.DEFAULT_EF_CONSTRUCTION

        self.index: faiss.IndexIDMap2 = faiss.IndexIDMap2(base)
        self.ef_search: int = 32  # default runtime ef, can be tuned dynamically

        self._stats = IndexStats(dim=dim)
        # simple in-memory cache for repeated queries
        self._cache: dict[tuple[tuple[float, ...], int, int], tuple[list[str], list[float]]] = {}
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        log.info("FAISS HNSW index initialised: dim=%d, metric=%s", dim, space)

    # ─────────────────────── Helpers ────────────────────────
    @staticmethod
    def _to_float32(arr: NDArray) -> NDArray:
        return arr.astype(np.float32, copy=False)

    def _string_to_int(self, s: str) -> int:
        if s in self._reverse_id_map:
            return self._reverse_id_map[s]
        int_id = len(self._id_map) + 1
        self._id_map[int_id] = s
        self._reverse_id_map[s] = int_id
        return int_id

    def _int_to_string(self, i: int) -> str:
        return self._id_map.get(i, hex(int(i)))

    # ─────────────────────── Mutators ────────────────────────
    def add_vectors(self, ids: Sequence[str], vectors: NDArray) -> None:
        """Add vectors with external string IDs."""
        if len(ids) != len(vectors):
            raise ANNIndexError("ids and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ANNIndexError(
                f"dimension mismatch: expected dim={self.dim}, got {vectors.shape[1]}"
            )

        dup = [item for item, cnt in Counter(ids).items() if cnt > 1]
        if dup:
            raise ANNIndexError("duplicate IDs in input")

        with self._lock:
            existing = {i for i in ids if i in self._reverse_id_map}
            if existing:
                raise ANNIndexError("IDs already present")

            vecs = self._to_float32(np.asarray(vectors))
            if self.space == "cosine":
                faiss.normalize_L2(vecs)
            id_arr = np.array([self._string_to_int(i) for i in ids], dtype="int64")
            self.index.add_with_ids(vecs, id_arr)

            self._stats.total_vectors += len(ids)
            _VEC_ADDED.inc(len(ids))
            log.debug("Added %d vectors", len(ids))
            self._cache.clear()

    def remove_ids(self, ids: Iterable[str]) -> None:
        int_ids = np.array([self._string_to_int(i) for i in ids], dtype="int64")
        selector = faiss.IDSelectorBatch(int_ids.size, faiss.swig_ptr(int_ids))
        with self._lock:
            removed = self.index.remove_ids(selector)
            self._stats.total_vectors -= int(removed)
            _VEC_DELETED.inc(int(removed))
            log.debug("Removed %d vectors", removed)
            if removed:
                for _id in int_ids:
                    sid = self._id_map.pop(int(_id), None)
                    if sid:
                        self._reverse_id_map.pop(sid, None)
                self._cache.clear()

    # ─────────────────────── Query ────────────────────────
    def search(
        self,
        vector: NDArray,
        *,
        k: int = 5,
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        if vector.shape[-1] != self.dim:
            raise ANNIndexError(
                f"dimension mismatch: expected dim={self.dim}, got {vector.shape[-1]}"
            )
            
        vec_flat = vector.flatten()
        key = (tuple(float(x) for x in vec_flat), k, ef_search or self.ef_search)
        if key in self._cache:
            return self._cache[key]
            
        vec = self._to_float32(vector.reshape(1, -1))
        if self.space == "cosine":
            faiss.normalize_L2(vec)

        if ef_search is not None:
            self.index.hnsw.efSearch = ef_search
            self.ef_search = ef_search

        start = perf_counter()
        try:
            with self._lock:
                distances, int_ids = self.index.search(vec, k)
        except Exception as exc:  # noqa: BLE001
            _QUERY_ERR.inc()
            raise ANNIndexError("FAISS search failed") from exc

        if int_ids.size == 0:
            self._cache[key] = ([], [])
            return [], []
        latency = (perf_counter() - start) * 1000.0

        self._stats.total_queries += 1
        self._stats.avg_latency_ms = (
            self._stats.avg_latency_ms * (self._stats.total_queries - 1) + latency
        ) / self._stats.total_queries
        _QUERY_CNT.inc()

        ids = [self._int_to_string(int(i)) for i in int_ids[0]]
        dists = list(distances[0])
        self._cache[key] = (ids, dists)
        return ids, dists

    # ─────────────────────── Rebuild / IO ────────────────────────
    def rebuild(self, vectors: NDArray, ids: Sequence[str]) -> None:
        """Recreate the FAISS index from scratch in a transactional way."""
        temp = FaissHNSWIndex(self.dim, space=self.space)
        temp.add_vectors(ids, vectors)
        with self._lock:
            self.index = temp.index
            self._id_map = temp._id_map
            self._reverse_id_map = temp._reverse_id_map
            self._stats.total_vectors = len(ids)
            self._stats.last_rebuild = perf_counter()
            self._cache.clear()
            log.info("Index rebuilt with %d vectors", len(ids))

    def save(self, path: str) -> None:
        with self._lock:
            faiss.write_index(self.index, path)
            (Path(path).with_suffix(".map.json")).write_text(json.dumps(self._id_map))
            log.info("Index saved to %s", path)

    def load(self, path: str) -> None:
        with self._lock:
            self.index = faiss.read_index(path)
            map_path = Path(path).with_suffix(".map.json")
            if map_path.exists():
                self._id_map = {int(k): v for k, v in json.loads(map_path.read_text()).items()}
                self._reverse_id_map = {v: int(k) for k, v in self._id_map.items()}
            self._stats.total_vectors = self.index.ntotal
            self._cache.clear()
            log.info("Index loaded from %s", path)

    # ─────────────────────── Info ────────────────────────
    def stats(self) -> IndexStats:
        return self._stats
