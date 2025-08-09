# index.py — FAISS‑based ANN index for Unified Memory System
# Version: v{__version__}

"""
Vector‑similarity index built on top of FAISS (IndexHNSWFlat) with
ID mapping, statistics, dynamic search tuning, and Prometheus hooks.

Example usage:
    from memory_system.core.index import FaissHNSWIndex
    idx = FaissHNSWIndex(dim=768)
    idx.add_vectors(["id‑1", "id‑2"], np.random.rand(2, 768))
    ids, dist = idx.search(np.random.rand(768), k=5)

Thread-safe: concurrent searches, exclusive writes.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
from numpy import ndarray as NDArray
from prometheus_client import Gauge, Histogram

from memory_system.utils.exceptions import StorageError
from memory_system.utils.metrics import prometheus_counter
from memory_system.utils.rwlock import RWLock

log = logging.getLogger(__name__)

# ───────────────────── Prometheus collectors ─────────────────────
_VEC_ADDED = prometheus_counter("ums_vectors_added_total", "Vectors added to ANN index")
_VEC_DELETED = prometheus_counter("ums_vectors_deleted_total", "Vectors deleted from ANN index")
_QUERY_CNT = prometheus_counter("ums_ann_queries_total", "ANN queries executed")
_QUERY_ERR = prometheus_counter("ums_ann_query_errors_total", "Errors while querying ANN index")
_QUERY_LATENCY = Histogram(
    "ums_ann_query_latency_seconds",
    "Latency of ANN index searches",
)
_INDEX_SIZE = Gauge(
    "ums_ann_index_size",
    "Number of vectors currently stored in the ANN index",
)


# ────────────────────────── Exceptions ───────────────────────────
class ANNIndexError(StorageError, ValueError):
    """Raised for duplicate IDs, dimension mismatch, or internal FAISS errors."""


# ─────────────────────────── Dataclass ────────────────────────────
@dataclass(slots=True)
class IndexStats:
    """Basic statistics about the FAISS index."""

    dim: int
    total_vectors: int = 0
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    last_rebuild: float | None = None
    extra: dict[str, int | float] = field(default_factory=dict)


# ────────────────────────── Main class ────────────────────────────
class FaissHNSWIndex:
    """High-level wrapper over FAISS indices with ID mapping and stats.

    Historically this class exposed only :class:`faiss.IndexHNSWFlat`. It now
    supports a small subset of alternative FAISS index types and can optionally
    move the index to GPU memory. Behaviour defaults to the previous HNSW CPU
    setup so existing deployments keep working out of the box.
    """

    DEFAULT_EF_CONSTRUCTION: int = int(os.getenv("UMS_EF_CONSTRUCTION", "128"))
    DEFAULT_HNSW_M: int = int(os.getenv("UMS_HNSW_M", "32"))
    DEFAULT_EF_SEARCH: int = int(os.getenv("UMS_EF_SEARCH", "32"))
    DEFAULT_INDEX_TYPE: str = os.getenv("UMS_INDEX_TYPE", "HNSW").upper()
    DEFAULT_USE_GPU: bool = os.getenv("UMS_USE_GPU", "0") == "1"
    DEFAULT_IVF_NLIST: int = int(os.getenv("UMS_IVF_NLIST", "100"))
    DEFAULT_IVF_NPROBE: int = int(os.getenv("UMS_IVF_NPROBE", "8"))
    DEFAULT_PQ_M: int = int(os.getenv("UMS_PQ_M", "16"))
    DEFAULT_PQ_BITS: int = int(os.getenv("UMS_PQ_BITS", "8"))

    def __init__(
        self,
        dim: int,
        *,
        ef_construction: int | None = None,
        M: int | None = None,
        ef_search: int | None = None,
        space: str = "cosine",
        index_type: str | None = None,
        use_gpu: bool | None = None,
        ivf_nlist: int | None = None,
        ivf_nprobe: int | None = None,
        pq_m: int | None = None,
        pq_bits: int | None = None,
    ) -> None:
        """Initialise the FAISS index wrapper."""
        self.dim = dim
        self.space = space
        self._lock = RWLock()
        self.index_type = (index_type or self.DEFAULT_INDEX_TYPE).upper()
        self.use_gpu = use_gpu if use_gpu is not None else self.DEFAULT_USE_GPU
        self.ivf_nlist = ivf_nlist or self.DEFAULT_IVF_NLIST
        self.pq_m = pq_m or self.DEFAULT_PQ_M
        self.pq_bits = pq_bits or self.DEFAULT_PQ_BITS

        # Build underlying FAISS index
        metric = faiss.METRIC_INNER_PRODUCT if space == "cosine" else faiss.METRIC_L2
        if self.index_type in {"IVF", "IVFFLAT"}:
            quantizer = faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            base = faiss.IndexIVFFlat(quantizer, dim, self.ivf_nlist, metric)
            self.ef_search = ivf_nprobe or self.DEFAULT_IVF_NPROBE
            base.nprobe = self.ef_search
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            base = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            self.ef_search = ivf_nprobe or self.DEFAULT_IVF_NPROBE
            base.nprobe = self.ef_search
        elif self.index_type == "HNSWPQ":
            base = faiss.IndexHNSWPQ(dim, self.pq_m, self.pq_bits, metric)
            base.hnsw.efConstruction = ef_construction or self.DEFAULT_EF_CONSTRUCTION
            self.ef_search = ef_search or self.DEFAULT_EF_SEARCH
            base.hnsw.efSearch = self.ef_search
        elif self.index_type == "OPQ":
            quantizer = faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            ivfpq = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            self.ef_search = ivf_nprobe or self.DEFAULT_IVF_NPROBE
            ivfpq.nprobe = self.ef_search
            opq = faiss.OPQMatrix(dim, self.pq_m)
            base = faiss.IndexPreTransform(opq, ivfpq)
        else:  # default to HNSW
            base = faiss.IndexHNSWFlat(dim, M or self.DEFAULT_HNSW_M, metric)
            base.hnsw.efConstruction = ef_construction or self.DEFAULT_EF_CONSTRUCTION
            self.ef_search = ef_search or self.DEFAULT_EF_SEARCH
            base.hnsw.efSearch = self.ef_search

        if self.use_gpu:
            try:
                base = faiss.index_cpu_to_all_gpus(base)
                log.info("FAISS index moved to GPU")
            except Exception:
                log.warning("Failed to move FAISS index to GPU; using CPU index")

        self.index: faiss.IndexIDMap2 = faiss.IndexIDMap2(base)

        self._stats = IndexStats(dim=dim)
        self._cache: dict[tuple[int, int, int], tuple[list[str], list[float]]] = {}
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        self._vectors: dict[int, NDArray] = {}
        self._warmed_up: bool = False
        _INDEX_SIZE.set(0)
        log.info("FAISS %s index initialised: dim=%d, metric=%s", self.index_type, dim, space)

    # ────────────────────────── Internal ──────────────────────────
    def _warm_up(self) -> None:
        """Run a dummy search to initialize FAISS structures.

        This is called lazily after the first set of vectors is stored so
        that FAISS does not allocate memory or compute graph structures
        until the index actually contains data.
        """
        if not self._warmed_up and self.index.ntotal > 0:
            dummy = np.asarray([[0.0] * self.dim], dtype=np.float32)
            if self.space == "cosine":
                faiss.normalize_L2(dummy)
            with self._lock.reader_lock():
                self.index.search(dummy, 1)
            self._warmed_up = True

    # ─────────────────────── Helpers ────────────────────────
    @staticmethod
    def _to_float32(arr: NDArray) -> NDArray:
        """Ensure array is float32 dtype."""
        return arr.astype(np.float32, copy=False)

    def _string_to_int(self, s: str) -> int:
        """Map string ID to int, creating mapping if needed."""
        if s in self._reverse_id_map:
            return self._reverse_id_map[s]
        int_id = len(self._id_map) + 1
        self._id_map[int_id] = s
        self._reverse_id_map[s] = int_id
        return int_id

    def _int_to_string(self, i: int) -> str:
        """Map int ID back to string, or hex if missing."""
        return self._id_map.get(i, hex(int(i)))

    def auto_tune(self, sample_vectors: NDArray) -> tuple[int, int, int]:
        """Benchmark several HNSW configurations and pick the best.

        Parameters
        ----------
        sample_vectors:
            Sample data used both for building the temporary indexes and as
            queries. It should be shaped ``(n, dim)``.

        Returns
        -------
        tuple[int, int, int]
            ``(M, ef_construction, ef_search)`` of the selected configuration.
        """

        if sample_vectors.size == 0:
            return (self.DEFAULT_HNSW_M, self.DEFAULT_EF_CONSTRUCTION, self.DEFAULT_EF_SEARCH)

        vecs = self._to_float32(np.asarray(sample_vectors))
        if vecs.shape[1] != self.dim:
            raise ANNIndexError(f"dimension mismatch: expected dim={self.dim}, got {vecs.shape[1]}")
        if self.space == "cosine":
            faiss.normalize_L2(vecs)

        metric = faiss.METRIC_INNER_PRODUCT if self.space == "cosine" else faiss.METRIC_L2

        flat = faiss.IndexFlatIP(self.dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
        flat.add(vecs)

        k = min(5, len(vecs))
        _, gt_idx = flat.search(vecs, k)

        configs = [
            (16, 100, 50),
            (32, 200, 100),
            (64, 400, 200),
        ]

        results: list[tuple[float, float, int, int, int]] = []
        for M, ef_c, ef_s in configs:
            idx = faiss.IndexHNSWFlat(self.dim, M, metric)
            idx.hnsw.efConstruction = ef_c
            idx.hnsw.efSearch = ef_s
            idx.add(vecs)

            start = perf_counter()
            _, I = idx.search(vecs, k)
            latency = (perf_counter() - start) * 1000.0 / len(vecs)

            recall = 0.0
            for q in range(len(vecs)):
                recall += len(set(I[q]) & set(gt_idx[q])) / k
            recall /= len(vecs)

            results.append((recall, latency, M, ef_c, ef_s))
            log.info(
                "Autotune candidate M=%d efC=%d efS=%d -> recall=%.3f latency=%.2fms",
                M,
                ef_c,
                ef_s,
                recall,
                latency,
            )

        results.sort(key=lambda x: (-x[0], x[1]))
        best = results[0]
        _, _, M, ef_c, ef_s = best
        log.info(
            "Autotune selected M=%d efC=%d efS=%d (recall=%.3f latency=%.2fms)",
            M,
            ef_c,
            ef_s,
            best[0],
            best[1],
        )
        return M, ef_c, ef_s

    # ─────────────────────── Mutators ────────────────────────
    def add_vectors(self, ids: Sequence[str], vectors: NDArray) -> None:
        """Add vectors with external string IDs."""
        if len(ids) != len(vectors):
            raise ANNIndexError("ids and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ANNIndexError(f"dimension mismatch: expected dim={self.dim}, got {vectors.shape[1]}")

        dup = [item for item, cnt in Counter(ids).items() if cnt > 1]
        if dup:
            raise ANNIndexError("duplicate IDs in input")

        with self._lock.writer_lock():
            existing = {i for i in ids if i in self._reverse_id_map}
            if existing:
                raise ANNIndexError("IDs already present")

            vecs = self._to_float32(np.asarray(vectors))
            if self.space == "cosine":
                faiss.normalize_L2(vecs)
            id_arr = np.array([self._string_to_int(i) for i in ids], dtype="int64")

            # ``IndexIDMap2`` reports itself as trained even if the underlying
            # FAISS index still requires training (e.g. IVF/PQ variants). This
            # previously meant that PQ-based indexes were left untrained and
            # calling ``add_with_ids`` would crash the interpreter with a
            # segmentation fault. To avoid this we explicitly inspect the base
            # index and train it when necessary before adding any vectors.
            base_index = faiss.downcast_index(self.index.index)
            # Some wrapper indexes (e.g. ``IndexPreTransform`` used by OPQ)
            # report themselves as trained even when the underlying IVF/PQ
            # index still requires training. ``extract_index_ivf`` walks
            # through such wrappers and returns the innermost ``IndexIVF``
            # instance when present so we can reliably check its state.
            ivf_index = faiss.extract_index_ivf(base_index)
            if (not base_index.is_trained) or (ivf_index and not ivf_index.is_trained):
                # Indices like IVF/PQ require training before adding vectors
                self.index.train(vecs)
            self.index.add_with_ids(vecs, id_arr)
            for idx, int_id in enumerate(id_arr):
                self._vectors[int(int_id)] = vecs[idx]

            self._stats.total_vectors += len(ids)
            _INDEX_SIZE.set(self._stats.total_vectors)
            _VEC_ADDED.inc(len(ids))
            log.debug("Added %d vectors", len(ids))
            self._cache.clear()

        # Warm up FAISS after inserting the first vectors so that
        # internal search structures are initialized lazily only when
        # we actually have data stored.
        self._warm_up()

    def add_vectors_streaming(
        self,
        iterator: Iterable[tuple[str, NDArray | Sequence[float]]],
        *,
        batch_size: int = 1000,
    ) -> None:
        """Add vectors from an iterator without loading all data at once."""

        ids: list[str] = []
        vecs: list[NDArray] = []
        for _id, vec in iterator:
            ids.append(_id)
            vecs.append(np.asarray(vec, dtype=np.float32))
            if len(ids) >= batch_size:
                self.add_vectors(ids, np.asarray(vecs, dtype=np.float32))
                ids.clear()
                vecs.clear()
        if ids:
            self.add_vectors(ids, np.asarray(vecs, dtype=np.float32))

    def _rebuild_from_vectors(self) -> None:
        """Reconstruct the FAISS index from vectors kept in memory."""
        metric = faiss.METRIC_INNER_PRODUCT if self.space == "cosine" else faiss.METRIC_L2
        if self.index_type in {"IVF", "IVFFLAT"}:
            quantizer = faiss.IndexFlatL2(self.dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dim)
            base = faiss.IndexIVFFlat(quantizer, self.dim, self.ivf_nlist, metric)
            base.nprobe = self.ef_search
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(self.dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dim)
            base = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            base.nprobe = self.ef_search
        elif self.index_type == "HNSWPQ":
            base = faiss.IndexHNSWPQ(self.dim, self.pq_m, self.pq_bits, metric)
            base.hnsw.efConstruction = self.DEFAULT_EF_CONSTRUCTION
            base.hnsw.efSearch = self.ef_search
        elif self.index_type == "OPQ":
            quantizer = faiss.IndexFlatL2(self.dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.dim)
            ivfpq = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            ivfpq.nprobe = self.ef_search
            opq = faiss.OPQMatrix(self.dim, self.pq_m)
            base = faiss.IndexPreTransform(opq, ivfpq)
        else:
            base = faiss.IndexHNSWFlat(self.dim, self.DEFAULT_HNSW_M, metric)
            base.hnsw.efConstruction = self.DEFAULT_EF_CONSTRUCTION
            base.hnsw.efSearch = self.ef_search

        if self.use_gpu:
            try:
                base = faiss.index_cpu_to_all_gpus(base)
            except Exception:
                log.warning("Failed to move rebuilt FAISS index to GPU; using CPU index")

        new_index = faiss.IndexIDMap2(base)
        if self._vectors:
            ids = np.array(list(self._vectors.keys()), dtype="int64")
            vecs = np.vstack(list(self._vectors.values())).astype("float32")
            base_index = faiss.downcast_index(new_index.index)
            if not base_index.is_trained:
                new_index.train(vecs)
            new_index.add_with_ids(vecs, ids)
        self.index = new_index
        self._stats.total_vectors = len(self._vectors)
        _INDEX_SIZE.set(self._stats.total_vectors)
        self._cache.clear()
        self._warm_up()

    def remove_ids(self, ids: Iterable[str]) -> None:
        """Remove vectors by string IDs."""
        int_ids = [self._reverse_id_map.get(i) for i in ids if i in self._reverse_id_map]
        if not int_ids:
            return
        arr = np.array(int_ids, dtype="int64")
        selector = faiss.IDSelectorBatch(arr.size, faiss.swig_ptr(arr))
        with self._lock.writer_lock():
            try:
                removed = self.index.remove_ids(selector)
            except RuntimeError:
                removed = 0
            if removed:
                _VEC_DELETED.inc(int(removed))
            for iid in arr:
                self._vectors.pop(int(iid), None)
                sid = self._id_map.pop(int(iid), None)
                if sid:
                    self._reverse_id_map.pop(sid, None)
            if removed == 0:
                self._rebuild_from_vectors()
            else:
                self._stats.total_vectors = len(self._vectors)
                _INDEX_SIZE.set(self._stats.total_vectors)
                self._cache.clear()
            if removed == 0:
                self._stats.total_vectors = len(self._vectors)
                _INDEX_SIZE.set(self._stats.total_vectors)

    # ─────────────────────── Query ────────────────────────
    def search(
        self,
        vector: NDArray,
        *,
        k: int = 5,
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Search for k nearest neighbors of a vector."""
        if vector.shape[-1] != self.dim:
            raise ANNIndexError(f"dimension mismatch: expected dim={self.dim}, got {vector.shape[-1]}")

        if self.index.ntotal == 0:
            return [], []

        vec32 = self._to_float32(np.asarray(vector))
        vec1d = vec32.flatten()
        vec_bytes = (
            vec1d.tobytes() if hasattr(vec1d, "tobytes") else struct.pack(f"{len(vec1d)}f", *[float(x) for x in vec1d])
        )
        key = (hash(vec_bytes), k, ef_search or self.ef_search)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vec = vec1d.reshape(1, -1)
        if self.space == "cosine":
            faiss.normalize_L2(vec)

        if ef_search is not None:
            if self.index_type in {"IVF", "IVFFLAT", "IVFPQ", "OPQ"}:
                faiss.extract_index_ivf(self.index).nprobe = ef_search
            else:
                # IndexIDMap2 does not expose the HNSW params directly
                faiss.downcast_index(self.index.index).hnsw.efSearch = ef_search
            self.ef_search = ef_search

        start = perf_counter()
        try:
            with self._lock.reader_lock():
                distances, int_ids = self.index.search(vec, k)
        except Exception as exc:
            _QUERY_ERR.inc()
            raise ANNIndexError("FAISS search failed") from exc

        if int_ids.size == 0:
            self._cache[key] = ([], [])
            return [], []
        latency_sec = perf_counter() - start
        latency = latency_sec * 1000.0
        _QUERY_LATENCY.observe(latency_sec)

        self._stats.total_queries += 1
        self._stats.avg_latency_ms = (
            self._stats.avg_latency_ms * (self._stats.total_queries - 1) + latency
        ) / self._stats.total_queries
        _QUERY_CNT.inc()

        ids = [self._int_to_string(int(i)) for i in int_ids[0]]
        dists = list(distances[0])
        self._cache[key] = (ids, dists)
        return ids, dists

    def get_vector(self, id: str) -> NDArray | None:
        """Return stored vector for ``id`` if present."""
        int_id = self._reverse_id_map.get(id)
        if int_id is None:
            return None
        vec = self._vectors.get(int_id)
        if vec is None:
            return None
        return vec.copy()

    # ─────────────────────── Rebuild / IO ────────────────────────
    def rebuild(self, vectors: NDArray, ids: Sequence[str]) -> bool:
        """Recreate the FAISS index from scratch in a transactional way. Returns True if successful."""
        temp = FaissHNSWIndex(
            self.dim,
            space=self.space,
            index_type=self.index_type,
            use_gpu=self.use_gpu,
            ivf_nlist=self.ivf_nlist,
            ivf_nprobe=self.ef_search,
            pq_m=self.pq_m,
            pq_bits=self.pq_bits,
            ef_search=self.ef_search,
        )
        try:
            temp.add_vectors(ids, vectors)
            with self._lock.writer_lock():
                self.index = temp.index
                self._id_map = temp._id_map
                self._reverse_id_map = temp._reverse_id_map
                self._stats.total_vectors = len(ids)
                _INDEX_SIZE.set(self._stats.total_vectors)
                self._stats.last_rebuild = perf_counter()
                self._cache.clear()
                log.info("Index rebuilt with %d vectors", len(ids))
            return True
        except Exception as e:
            log.error(f"Index rebuild failed: {e}")
            return False

    def save(self, path: str) -> None:
        """Save index and ID map to disk. Logs error if fails."""
        try:
            with self._lock.writer_lock():
                faiss.write_index(self.index, path)
                (Path(path).with_suffix(".map.json")).write_text(json.dumps(self._id_map))
                log.info("Index saved to %s", path)
        except Exception as e:
            log.error(f"Failed to save index: {e}")

    def load(self, path: str) -> None:
        """Load index and ID map from disk. Logs error if fails."""
        try:
            with self._lock.writer_lock():
                self.index = faiss.read_index(path)
                map_path = Path(path).with_suffix(".map.json")
                if map_path.exists():
                    self._id_map = {int(k): v for k, v in json.loads(map_path.read_text()).items()}
                    self._reverse_id_map = {v: int(k) for k, v in self._id_map.items()}
                self._stats.total_vectors = self.index.ntotal
                _INDEX_SIZE.set(self._stats.total_vectors)
                self._cache.clear()
                log.info("Index loaded from %s", path)
        except Exception as e:
            log.error(f"Failed to load index: {e}")

    # ─────────────────────── Info ────────────────────────
    def stats(self) -> IndexStats:
        """Return current index statistics."""
        return self._stats


class MultiModalFaissIndex:
    """Manage separate FAISS indices for multiple modalities.

    All keyword arguments are forwarded to :class:`FaissHNSWIndex`, enabling
    access to any underlying FAISS option such as IVF, PQ, GPU, etc.
    """

    def __init__(self, vector_dims: dict[str, int], **faiss_kwargs) -> None:
        self._indices: dict[str, FaissHNSWIndex] = {}
        for mod, dim in vector_dims.items():
            self._indices[mod] = FaissHNSWIndex(dim=dim, **faiss_kwargs)

    # Basic routing operations -------------------------------------------------
    def add_vectors(self, modality: str, ids: list[str], vectors: NDArray, **kwargs: Any) -> None:
        self._indices[modality].add_vectors(ids, vectors, **kwargs)

    def search(
        self,
        modality: str,
        vector: NDArray,
        *,
        k: int = 5,
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        return self._indices[modality].search(vector, k=k, ef_search=ef_search)

    # Persistence --------------------------------------------------------------
    def save(self, path: str) -> None:
        base = Path(path)
        for mod, idx in self._indices.items():
            idx.save(str(base.with_suffix(base.suffix + f".{mod}")))

    def load(self, path: str) -> None:
        base = Path(path)
        for mod, idx in self._indices.items():
            p = base.with_suffix(base.suffix + f".{mod}")
            if p.exists():
                idx.load(str(p))

    # Stats -------------------------------------------------------------------
    def stats(self, modality: str | None = None) -> IndexStats:
        if modality is not None:
            return self._indices[modality].stats()
        total = sum(idx.stats().total_vectors for idx in self._indices.values())
        dim = next(iter(self._indices.values())).stats().dim if self._indices else 0
        return IndexStats(dim=dim, total_vectors=total)
