"""
Fallback FAISS index implemented with NumPy.

This module provides a tiny, pure Python stand-in for the FAISS-based
``FaissHNSWIndex`` used throughout the project.  It supports only the minimal
subset of features exercised in the test-suite and is intended purely as a
compatibility layer when the real FAISS bindings are unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Sequence

    from numpy import ndarray

from memory_system.utils.metrics import Gauge, Histogram, prometheus_counter

# Prometheus metric stubs used in tests
_VEC_ADDED = prometheus_counter("ums_vectors_added_total", "Vectors added to ANN index")
_VEC_DELETED = prometheus_counter("ums_vectors_deleted_total", "Vectors deleted from ANN index")
_QUERY_CNT = prometheus_counter("ums_ann_queries_total", "ANN queries executed")
_QUERY_ERR = prometheus_counter("ums_ann_query_errors_total", "Errors while querying ANN index")
_QUERY_LATENCY = Histogram("ums_ann_query_latency_seconds", "Latency of ANN index searches")
_INDEX_SIZE = Gauge("ums_ann_index_size", "Number of vectors currently stored in the ANN index")

log = logging.getLogger(__name__)


class ANNIndexError(ValueError):
    """Simplified ANN index error used by the fallback implementation."""


@dataclass(slots=True)
class IndexStats:
    dim: int
    total_vectors: int = 0
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    last_rebuild: float | None = None
    extra: dict[str, int | float] = field(default_factory=dict)


class FaissHNSWIndex:
    """
    Numpy-based stand in for :class:`faiss.IndexHNSWFlat`.

    The real project exposes a rich FAISS wrapper.  For environments where the
    compiled FAISS bindings are unavailable we provide a tiny, slow in-memory
    implementation that mimics the parts of the API exercised in the tests.  It
    performs brute-force cosine similarity searches and persists data using
    ``numpy.save``.
    """

    DEFAULT_EF_CONSTRUCTION: int = 200
    DEFAULT_HNSW_M: int = 32
    DEFAULT_EF_SEARCH: int = 32
    DEFAULT_INDEX_TYPE: str = "HNSW"
    DEFAULT_USE_GPU: bool = False
    DEFAULT_IVF_NLIST: int = 1
    DEFAULT_IVF_NPROBE: int = 1
    MAX_IVF_NLIST: int = 1_000_000
    DEFAULT_PQ_M: int = 16
    DEFAULT_PQ_BITS: int = 8

    def __init__(
        self,
        dim: int,
        *,
        space: str = "cosine",
        ef_search: int | None = None,
        ivf_nlist: int | None = None,
        ivf_nprobe: int | None = None,
        **_: object,
    ) -> None:
        self.dim = dim
        self.space = space
        self.ivf_nlist = ivf_nlist or self.DEFAULT_IVF_NLIST
        if self.ivf_nlist < 1:
            log.warning("ivf_nlist=%d below 1; using 1", self.ivf_nlist)
            self.ivf_nlist = 1
        elif self.ivf_nlist > self.MAX_IVF_NLIST:
            log.warning(
                "ivf_nlist=%d above max %d; using %d",
                self.ivf_nlist,
                self.MAX_IVF_NLIST,
                self.MAX_IVF_NLIST,
            )
            self.ivf_nlist = self.MAX_IVF_NLIST

        self.nprobe = ivf_nprobe or self.DEFAULT_IVF_NPROBE
        if self.nprobe < 1:
            log.warning("ivf_nprobe=%d below 1; using 1", self.nprobe)
            self.nprobe = 1
        elif self.nprobe > self.ivf_nlist:
            log.warning(
                "ivf_nprobe=%d exceeds ivf_nlist=%d; using %d",
                self.nprobe,
                self.ivf_nlist,
                self.ivf_nlist,
            )
            self.nprobe = self.ivf_nlist

        self.ef_search = ef_search or self.DEFAULT_EF_SEARCH
        self.index: dict[str, ndarray] = {}
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        self._stats = IndexStats(dim=dim)

    # ------------------------------------------------------------------ helpers
    def _normalise(self, arr: ndarray) -> ndarray:
        if self.space == "cosine":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr.astype("float32")

    # ---------------------------------------------------------------- operations
    def add_vectors(self, ids: Sequence[str], vectors: ndarray) -> None:
        vecs = self._normalise(np.asarray(vectors, dtype="float32"))
        if vecs.shape[1] != self.dim:
            raise ANNIndexError("dimension mismatch")
        if len(ids) != len(vecs):
            raise ANNIndexError("length mismatch")
        if len(set(ids)) != len(ids):
            raise ANNIndexError("duplicate ids")
        existing = set(ids) & set(self._reverse_id_map)
        if existing:
            raise ANNIndexError("ids already present")
        start = len(self._id_map) + 1
        for offset, (sid, vec) in enumerate(zip(ids, vecs, strict=False)):
            int_id = start + offset
            self._id_map[int_id] = sid
            self._reverse_id_map[sid] = int_id
            self.index[sid] = vec
        self._stats.total_vectors = len(self.index)

    def search(self, vector: ndarray, k: int) -> tuple[list[str], list[float]]:
        if np.asarray(vector).shape[0] != self.dim:
            raise ANNIndexError("dimension mismatch")
        q = self._normalise(np.asarray([vector], dtype="float32"))[0]
        sims: list[tuple[float, str]] = []
        for sid, vec in self.index.items():
            sims.append((float(np.dot(q, vec)), sid))
        sims.sort(key=lambda x: x[0], reverse=True)
        ids = [sid for _, sid in sims[:k]]
        dists = [score for score, _ in sims[:k]]
        self._stats.total_queries += 1
        return ids, dists

    def remove_ids(self, ids: Sequence[str]) -> None:
        for sid in ids:
            if sid in self._reverse_id_map:
                self.index.pop(sid, None)
                iid = self._reverse_id_map.pop(sid)
                self._id_map.pop(iid, None)
        self._stats.total_vectors = len(self.index)

    def rebuild(self, vectors: ndarray, ids: Sequence[str]) -> None:
        self.index.clear()
        self._id_map.clear()
        self._reverse_id_map.clear()
        self.add_vectors(list(ids), np.asarray(vectors, dtype="float32"))
        self._stats.last_rebuild = perf_counter_ns() / 1e9

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        arr = np.stack([self.index[sid] for sid in self._reverse_id_map])
        saver = getattr(np, "save", None)
        if callable(saver):
            saver(path, arr)
        else:  # pragma: no cover - stub fallback
            Path(path).write_text(json.dumps(arr.tolist()))
        Path(path).with_suffix(".map.json").write_text(json.dumps(self._id_map))

    def load(self, path: str) -> None:
        if hasattr(np, "load"):
            arr = np.load(path)
        else:  # pragma: no cover - stub fallback
            arr = np.asarray(json.loads(Path(path).read_text()), dtype="float32")
        mapping = json.loads(Path(path).with_suffix(".map.json").read_text())
        self.index = {}
        self._id_map = {int(k): v for k, v in mapping.items()}
        self._reverse_id_map = {v: int(k) for k, v in self._id_map.items()}
        for int_id in sorted(self._id_map):
            sid = self._id_map[int_id]
            self.index[sid] = arr[int_id - 1].astype("float32")
        self._stats.total_vectors = len(self.index)

    # -------------------------------------------------------------------- stats
    def stats(self) -> IndexStats:
        return self._stats

    # ------------------------------------------------------------------ utility
    def auto_tune(self, sample: ndarray) -> tuple[int, int, int]:
        """Return placeholder tuning parameters."""
        return (
            self.DEFAULT_HNSW_M,
            self.DEFAULT_EF_CONSTRUCTION,
            self.DEFAULT_EF_SEARCH,
        )


class MultiModalFaissIndex:
    """Minimal multi-modal wrapper used when FAISS is unavailable."""

    def __init__(self, specs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._indices: dict[str, FaissHNSWIndex] = {}
        if specs:
            for mod, spec in specs.items():
                if isinstance(spec, dict):
                    dim = int(spec.get("dim", 0))
                    params = {k: v for k, v in spec.items() if k != "dim"}
                else:
                    dim = int(spec)
                    params = {}
                params.update(kwargs)
                self._indices[mod] = FaissHNSWIndex(dim, **params)

    def add_vectors(self, modality: str, ids: Sequence[str], vectors: ndarray) -> None:
        self._indices[modality].add_vectors(ids, vectors)

    def search(self, modality: str, vector: ndarray, k: int = 5) -> tuple[list[str], list[float]]:
        return self._indices[modality].search(vector, k)

    def auto_tune(self, samples: dict[str, ndarray]) -> dict[str, tuple[int, int, int]]:
        """Return per-modality tuning parameters.

        The fallback returns default parameters for each modality.
        """
        return {mod: idx.auto_tune(samples.get(mod, np.empty((0, idx.dim), dtype="float32"))) for mod, idx in self._indices.items()}

    def save(self, path: str) -> None:
        meta: dict[str, Any] = {mod: {"dim": idx.dim} for mod, idx in self._indices.items()}
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)
        for mod, idx in self._indices.items():
            idx.save(str(base / f"{mod}.npy"))
        (base / "manifest.json").write_text(json.dumps(meta))

    def load(self, path: str) -> None:
        base = Path(path)
        manifest = json.loads((base / "manifest.json").read_text())
        self._indices = {}
        for mod, spec in manifest.items():
            idx = FaissHNSWIndex(int(spec.get("dim", 0)))
            idx.load(str(base / f"{mod}.npy"))
            self._indices[mod] = idx

    def remove_ids(self, ids: Sequence[str]) -> None:
        for idx in self._indices.values():
            idx.remove_ids(ids)

    def rebuild(self, modality: str, vectors: ndarray, ids: Sequence[str]) -> None:
        self._indices[modality].rebuild(vectors, ids)

    def stats(self, modality: str | None = None) -> IndexStats:
        if modality is not None:
            return self._indices[modality].stats()
        total = sum(idx.stats().total_vectors for idx in self._indices.values())
        dim = next(iter(self._indices.values())).dim if self._indices else 0
        return IndexStats(dim=dim, total_vectors=total)


__all__ = [
    "_INDEX_SIZE",
    "_QUERY_CNT",
    "_QUERY_ERR",
    "_QUERY_LATENCY",
    "_VEC_ADDED",
    "_VEC_DELETED",
    "ANNIndexError",
    "FaissHNSWIndex",
    "IndexStats",
    "MultiModalFaissIndex",
]

