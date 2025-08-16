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

import asyncio
import json
import logging
import struct
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, cast

from numpy.typing import NDArray

from memory_system.utils.dependencies import require_faiss, require_numpy
from memory_system.utils.loop import get_loop

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import faiss
else:
    faiss = require_faiss()

from memory_system.utils.exceptions import StorageError
from memory_system.utils.metrics import Gauge, Histogram, prometheus_counter
from memory_system.utils.rwlock import RWLock
from memory_system.utils.security import CryptoContext

log = logging.getLogger(__name__)


def _faiss_has_gpu() -> bool:
    """Return ``True`` if the :mod:`faiss` module has GPU support."""
    try:
        return bool(faiss.get_num_gpus())
    except Exception:
        return False


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
    """
    High-level wrapper over FAISS indices with ID mapping and stats.

    Historically this class exposed only :class:`faiss.IndexHNSWFlat`. It now
    supports a small subset of alternative FAISS index types and can optionally
    move the index to GPU memory. Behaviour defaults to the previous HNSW CPU
    setup so existing deployments keep working out of the box.
    """

    DEFAULT_EF_CONSTRUCTION: int = 200
    DEFAULT_HNSW_M: int = 32
    DEFAULT_EF_SEARCH: int = 100
    DEFAULT_INDEX_TYPE: str = "HNSW"
    DEFAULT_USE_GPU: bool = False
    DEFAULT_IVF_NLIST: int = 100
    DEFAULT_IVF_NPROBE: int = 8
    MAX_IVF_NLIST: int = 1_000_000
    DEFAULT_PQ_M: int = 16
    DEFAULT_PQ_BITS: int = 8
    # FAISS PQ implementations become unstable when the dimensionality of the
    # sub‑vectors drops below four.  Use a named constant to keep the magic
    # number in one place and clarify intent when validating ``pq_m``.
    MIN_PQ_SUBVECTOR_DIM: int = 4

    @classmethod
    def _normalise_pq_m(cls, dim: int, pq_m: int) -> int:
        """
        Return a ``pq_m`` compatible with ``dim``.

        FAISS may segfault when the product quantisation parameter ``m`` does
        not evenly divide the dimensionality of the vectors or when the
        resulting sub-vector size becomes too small.  This helper adjusts the
        requested ``pq_m`` to the largest safe divisor, guaranteeing that each
        sub-vector has at least ``MIN_PQ_SUBVECTOR_DIM`` dimensions.
        """
        if dim % pq_m == 0 and dim // pq_m >= cls.MIN_PQ_SUBVECTOR_DIM:
            return pq_m
        for m in range(min(pq_m, dim), 0, -1):
            if dim % m == 0 and dim // m >= cls.MIN_PQ_SUBVECTOR_DIM:
                return m
        raise ANNIndexError(f"invalid pq_m={pq_m} for dim={dim}")

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
        crypto: CryptoContext | None = None,
        encrypt: bool = True,
    ) -> None:
        """Initialise the FAISS index wrapper."""
        self.dim = dim
        self.space = space
        self._lock = RWLock()
        self.index_type = (index_type or self.DEFAULT_INDEX_TYPE).upper()
        self.use_gpu = use_gpu if use_gpu is not None else self.DEFAULT_USE_GPU
        if self.use_gpu and not _faiss_has_gpu():
            log.warning("faiss-gpu not installed; forcing CPU mode")
            self.use_gpu = False
        self.M = M or self.DEFAULT_HNSW_M
        self.ef_construction = ef_construction or self.DEFAULT_EF_CONSTRUCTION
        self.ef_search = ef_search or self.DEFAULT_EF_SEARCH
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
        self.pq_m = pq_m or self.DEFAULT_PQ_M
        self.pq_bits = pq_bits or self.DEFAULT_PQ_BITS

        # ``faiss`` can segfault if the product quantisation parameter ``m``
        # does not evenly divide the dimensionality of the vectors or if the
        # resulting sub-vector size is too small.  Instead of blindly trusting
        # the caller we normalise ``pq_m`` here and, when necessary, fall back
        # to the largest safe divisor.  This mirrors the behaviour of other
        # FAISS bindings and prevents hard crashes for tiny dimensions such as
        # those used in the test-suite (e.g. dim=32 with the default m=16).
        if self.index_type in {"IVFPQ", "HNSWPQ", "OPQ"}:
            self.pq_m = self._normalise_pq_m(dim, self.pq_m)

        # Encryption context
        self._crypto: CryptoContext | None = crypto if encrypt else None
        if encrypt and self._crypto is None:
            try:
                self._crypto = CryptoContext.from_env()
            except RuntimeError:
                self._crypto = None

        # Build underlying FAISS index
        metric = faiss.METRIC_INNER_PRODUCT if space == "cosine" else faiss.METRIC_L2
        if self.index_type in {"IVF", "IVFFLAT"}:
            quantizer = (
                faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            )
            base = faiss.IndexIVFFlat(quantizer, dim, self.ivf_nlist, metric)
            self.nprobe = min(self.nprobe, self.ivf_nlist)
            self.ef_search = self.nprobe
            base.nprobe = self.nprobe
        elif self.index_type == "IVFPQ":
            quantizer = (
                faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            )
            base = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            self.nprobe = min(self.nprobe, self.ivf_nlist)
            self.ef_search = self.nprobe
            base.nprobe = self.nprobe
        elif self.index_type == "HNSWPQ":
            base = faiss.IndexHNSWPQ(dim, self.pq_m, self.pq_bits, metric)
            base.hnsw.efConstruction = self.ef_construction
            base.hnsw.efSearch = self.ef_search
        elif self.index_type == "OPQ":
            quantizer = (
                faiss.IndexFlatL2(dim) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(dim)
            )
            ivfpq = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            self.nprobe = min(self.nprobe, self.ivf_nlist)
            self.ef_search = self.nprobe
            ivfpq.nprobe = self.nprobe
            opq = faiss.OPQMatrix(dim, self.pq_m)
            base = faiss.IndexPreTransform(opq, ivfpq)
        else:  # default to HNSW
            base = faiss.IndexHNSWFlat(dim, self.M, metric)
            base.hnsw.efConstruction = self.ef_construction
            base.hnsw.efSearch = self.ef_search

        if self.use_gpu:
            try:
                base = faiss.index_cpu_to_all_gpus(base)
                log.info("FAISS index moved to GPU")
            except RuntimeError:
                log.warning("Failed to move FAISS index to GPU; using CPU index")

        self.index: faiss.IndexIDMap2 = faiss.IndexIDMap2(base)

        self._stats = IndexStats(dim=dim)
        self._cache: dict[tuple[int, int, int], tuple[list[str], list[float]]] = {}
        self._id_map: dict[int, str] = {}
        self._reverse_id_map: dict[str, int] = {}
        self._vectors: dict[int, NDArray[Any]] = {}
        self._warmed_up: bool = False
        _INDEX_SIZE.set(0)
        log.info(
            "FAISS %s index initialised: dim=%d metric=%s M=%d efC=%d efS=%d nlist=%d nprobe=%d pq_m=%d pq_bits=%d",
            self.index_type,
            dim,
            space,
            self.M,
            self.ef_construction,
            self.ef_search,
            self.ivf_nlist,
            self.nprobe,
            self.pq_m,
            self.pq_bits,
        )

    # ────────────────────────── Internal ──────────────────────────
    def _warm_up(self) -> None:
        """
        Run a dummy search to initialize FAISS structures.

        This is called lazily after the first set of vectors is stored so
        that FAISS does not allocate memory or compute graph structures
        until the index actually contains data.
        """
        np = require_numpy()
        if not self._warmed_up and self.index.ntotal > 0:
            dummy = np.asarray([[0.0] * self.dim], dtype=np.float32)
            if self.space == "cosine":
                faiss.normalize_L2(dummy)
            with self._lock.reader_lock():
                self.index.search(dummy, 1)
            self._warmed_up = True

    # ─────────────────────── Helpers ────────────────────────
    @staticmethod
    def _to_float32(arr: NDArray[Any]) -> NDArray[Any]:
        """Ensure array is float32 dtype."""
        np = require_numpy()
        # Use ``asarray`` to avoid relying on ``astype(copy=...)`` which some
        # NumPy stubs do not implement.
        return np.asarray(arr, dtype=np.float32)

    def _faiss_carray(
        self, arr: NDArray[Any] | Sequence[Any], *, ids: bool = False
    ) -> NDArray[Any]:
        """
        Return a safe C-contiguous array for consumption by FAISS.

        ``faiss`` operates on raw memory buffers without increasing Python
        reference counts.  If a caller passes a non owning view, FAISS may keep
        a pointer to memory that gets freed once the view goes out of scope,
        resulting in a segmentation fault.  Converting the input to an explicit
        C-ordered copy ensures the resulting array owns its data for the
        duration of the call.
        """
        np = require_numpy()
        if ids:
            return np.asarray(arr, dtype=np.int64).copy()
        return np.asarray(self._to_float32(np.asarray(arr)), dtype=np.float32).copy()

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

    # ─────────────────────── Training helpers ───────────────────────
    def _ensure_index_trained(self, vecs: NDArray[Any]) -> None:
        """
        Ensure the underlying FAISS index is trained before insertion.

        Some FAISS wrappers – notably :class:`IndexIDMap2` and
        :class:`IndexPreTransform` – may report themselves as trained even
        when the actual IVF/PQ structures still require training.  Calling
        ``add_with_ids`` on such an index results in a hard crash deep
        inside FAISS.  This helper performs a best‑effort inspection of the
        wrapped index and triggers training via the public ``train`` method
        when necessary.
        """
        # If the index already contains data we assume training has happened.
        # FAISS crashes with a segmentation fault if ``add_with_ids`` is
        # called on an un‑trained IVF/PQ based index.  Some wrappers (notably
        # :class:`IndexIDMap2` and :class:`IndexPreTransform`) may report
        # themselves as trained even when their inner IVF/PQ components are
        # not.  To avoid relying on fragile ``is_trained`` heuristics we train
        # on the first insert whenever the index type requires it.

        if self.index.ntotal > 0:
            return

        if self.index_type in {"IVF", "IVFFLAT", "IVFPQ", "HNSWPQ", "OPQ"}:
            np = require_numpy()
            # ``faiss`` expects training data to own its memory and be laid out
            # in C order.  ``_faiss_carray`` normalises the dtype and forces a
            # copy, protecting us from segmentation faults when callers supply
            # views or temporary arrays.
            train_vecs = self._faiss_carray(vecs)

            # Many PQ/IVF based FAISS indexes expect a reasonably large
            # training set.  Supplying too few vectors can leave the index in
            # an undefined state and later calls to ``add_with_ids`` may crash
            # the interpreter.  Replicate the data so that training always sees
            # at least ``min_train`` samples.
            #
            # ``IndexIVFPQ`` and related structures build ``m`` separate
            # codebooks each with ``2**pq_bits`` centroids.  Training with fewer
            # vectors than there are total centroids can leave the internal
            # state inconsistent which, in turn, used to manifest as a
            # segmentation fault when ``add_with_ids`` was invoked.  A mere 256
            # vectors were previously used which is insufficient for some
            # configurations (e.g. ``m=8`` with ``pq_bits=8`` requires 2048
            # vectors).  Scale ``min_train`` accordingly so that training always
            # operates on a sufficiently large sample size.
            min_train = max(256, self.pq_m * (1 << self.pq_bits))
            if train_vecs.shape[0] < min_train:
                reps = int(np.ceil(min_train / train_vecs.shape[0]))
                train_vecs = self._faiss_carray(np.tile(train_vecs, (reps, 1))[:min_train])

            if self.space == "cosine":
                faiss.normalize_L2(train_vecs)

            def _is_trained(idx: faiss.Index) -> bool:
                """Recursively verify that ``idx`` and all inner components are trained."""
                # ``IndexIDMap``/``IndexIDMap2`` simply forward calls to the
                # wrapped index but expose their own ``is_trained`` attribute.
                # That attribute often reports ``True`` even when the inner
                # IVF/PQ structures are untrained which used to make
                # ``add_with_ids`` crash with a segmentation fault.  We unwrap
                # such containers explicitly so the actual index is inspected.
                if isinstance(idx, faiss.IndexIDMap | faiss.IndexIDMap2):
                    return _is_trained(idx.index)

                if hasattr(idx, "is_trained") and not idx.is_trained:
                    return False

                # ``IndexPreTransform`` wraps a chain of preprocessing
                # transforms and another index.  Each element in the chain
                # (e.g. ``OPQMatrix``) may require training.  The previous
                # implementation only inspected the wrapped index which meant
                # untrained transforms could slip through and ``add_with_ids``
                # would later segfault.  Walk the transform chain explicitly
                # and verify every component reports itself as trained before
                # recursing into the inner index.
                if isinstance(idx, faiss.IndexPreTransform):
                    if hasattr(idx, "chain"):
                        chain = idx.chain
                        for i in range(chain.size()):
                            tr = chain.at(i)
                            if hasattr(tr, "is_trained") and not tr.is_trained:
                                return False
                    return _is_trained(idx.index)

                # IVF based indexes require the coarse quantizer and PQ (if
                # present) to be trained as well.
                if isinstance(idx, faiss.IndexIVF):
                    if not idx.is_trained:
                        return False
                    if hasattr(idx, "pq") and not idx.pq.is_trained:
                        return False

                return not (hasattr(idx, "pq") and not idx.pq.is_trained)

            # Attempt training; FAISS will raise on obvious problems such as
            # insufficient data.  Some wrapper classes (notably ``IndexIDMap``
            # and ``IndexPreTransform``) don't always forward ``train`` to the
            # contained index in older FAISS releases which could leave inner
            # structures untrained and later cause a hard crash in
            # ``add_with_ids``.  Train all nested indexes recursively to guard
            # against incomplete training.

            def _train_all(idx: faiss.Index, vecs: NDArray[Any]) -> None:
                """
                Recursively train ``idx`` and any wrapped index.

                ``faiss`` may report an index as trained even though some of
                its nested components still require training which would make a
                subsequent ``add_with_ids`` call segfault.  This helper makes
                training idempotent and guarantees that every inner index sees
                data in the representation it expects.
                """
                vecs = self._faiss_carray(vecs)
                # Skip work if this particular index already reports itself as
                # trained.  Double training is not only wasteful but has been
                # observed to trigger crashes for certain PQ based indexes.
                if _is_trained(idx):
                    return

                idx.train(vecs)
                inner = getattr(idx, "index", None)
                if not isinstance(inner, faiss.Index):
                    return

                if isinstance(idx, faiss.IndexPreTransform) and hasattr(idx, "chain"):
                    chain = idx.chain
                    transformed = vecs
                    for i in range(chain.size()):
                        tr = chain.at(i)
                        if hasattr(tr, "apply_py"):
                            transformed = tr.apply_py(transformed)
                    transformed = self._faiss_carray(transformed)
                    _train_all(inner, transformed)
                else:
                    _train_all(inner, vecs)

            _train_all(self.index, train_vecs)

            if not _is_trained(self.index):
                # Final attempt – if training still doesn't stick we raise an
                # explicit error rather than letting FAISS segfault.
                _train_all(self.index, train_vecs)
                if not _is_trained(self.index):
                    raise ANNIndexError("FAISS index failed to train")

    def auto_tune(self, sample_vectors: NDArray[Any]) -> tuple[int, int, int]:
        """
        Benchmark several HNSW configurations and pick the best.

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
        np = require_numpy()
        if sample_vectors.size == 0:
            return (self.DEFAULT_HNSW_M, self.DEFAULT_EF_CONSTRUCTION, self.DEFAULT_EF_SEARCH)
        vecs = self._to_float32(np.asarray(sample_vectors))
        if vecs.shape[1] != self.dim:
            raise ANNIndexError(f"dimension mismatch: expected dim={self.dim}, got {vecs.shape[1]}")
        if self.space == "cosine":
            faiss.normalize_L2(vecs)

        metric = faiss.METRIC_INNER_PRODUCT if self.space == "cosine" else faiss.METRIC_L2

        flat = (
            faiss.IndexFlatIP(self.dim)
            if metric == faiss.METRIC_INNER_PRODUCT
            else faiss.IndexFlatL2(self.dim)
        )
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

            start = perf_counter_ns()
            _, I = idx.search(vecs, k)
            latency = (perf_counter_ns() - start) / 1e6 / len(vecs)

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
    def add_vectors(self, ids: Sequence[str], vectors: NDArray[Any]) -> None:
        """Add vectors with external string IDs."""
        require_numpy()
        if len(ids) != len(vectors):
            raise ANNIndexError("ids and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ANNIndexError(
                f"dimension mismatch: expected dim={self.dim}, got {vectors.shape[1]}"
            )

        dup = [item for item, cnt in Counter(ids).items() if cnt > 1]
        if dup:
            raise ANNIndexError("duplicate IDs in input")

        with self._lock.writer_lock():
            existing = {i for i in ids if i in self._reverse_id_map}
            if existing:
                raise ANNIndexError("IDs already present")

            # Convert both vectors and ids to safe C-contiguous arrays before
            # handing them off to FAISS.  ``_faiss_carray`` ensures the
            # resulting arrays own their memory, preventing segmentation faults
            # when numpy views are used as input.
            vecs = self._faiss_carray(vectors)
            if self.space == "cosine":
                faiss.normalize_L2(vecs)
            id_arr = self._faiss_carray([self._string_to_int(i) for i in ids], ids=True)
            self._ensure_index_trained(vecs)
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
        iterator: Iterable[tuple[str, NDArray[Any] | Sequence[float]]],
        *,
        batch_size: int = 1000,
    ) -> None:
        """Add vectors from an iterator without loading all data at once."""
        np = require_numpy()
        ids: list[str] = []
        vecs: list[NDArray[Any]] = []
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
        require_numpy()
        metric = faiss.METRIC_INNER_PRODUCT if self.space == "cosine" else faiss.METRIC_L2
        if self.index_type in {"IVF", "IVFFLAT"}:
            quantizer = (
                faiss.IndexFlatL2(self.dim)
                if metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(self.dim)
            )
            base = faiss.IndexIVFFlat(quantizer, self.dim, self.ivf_nlist, metric)
            base.nprobe = min(self.ef_search, self.ivf_nlist)
        elif self.index_type == "IVFPQ":
            quantizer = (
                faiss.IndexFlatL2(self.dim)
                if metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(self.dim)
            )
            base = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            base.nprobe = min(self.ef_search, self.ivf_nlist)
        elif self.index_type == "HNSWPQ":
            base = faiss.IndexHNSWPQ(self.dim, self.pq_m, self.pq_bits, metric)
            base.hnsw.efConstruction = self.DEFAULT_EF_CONSTRUCTION
            base.hnsw.efSearch = self.ef_search
        elif self.index_type == "OPQ":
            quantizer = (
                faiss.IndexFlatL2(self.dim)
                if metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(self.dim)
            )
            ivfpq = faiss.IndexIVFPQ(
                quantizer,
                self.dim,
                self.ivf_nlist,
                self.pq_m,
                self.pq_bits,
                metric,
            )
            ivfpq.nprobe = min(self.ef_search, self.ivf_nlist)
            opq = faiss.OPQMatrix(self.dim, self.pq_m)
            base = faiss.IndexPreTransform(opq, ivfpq)
        else:
            base = faiss.IndexHNSWFlat(self.dim, self.DEFAULT_HNSW_M, metric)
            base.hnsw.efConstruction = self.DEFAULT_EF_CONSTRUCTION
            base.hnsw.efSearch = self.ef_search

        if self.use_gpu:
            try:
                base = faiss.index_cpu_to_all_gpus(base)
            except RuntimeError:
                log.warning("Failed to move rebuilt FAISS index to GPU; using CPU index")

        new_index = faiss.IndexIDMap2(base)
        if self._vectors:
            ids = self._faiss_carray(list(self._vectors.keys()), ids=True)
            vecs = self._faiss_carray(list(self._vectors.values()))
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
        require_numpy()
        int_ids = [self._reverse_id_map.get(i) for i in ids if i in self._reverse_id_map]
        if not int_ids:
            return
        arr = self._faiss_carray(int_ids, ids=True)
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
        vector: NDArray[Any],
        *,
        k: int = 5,
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Search for k nearest neighbors of a vector."""
        np = require_numpy()
        if vector.shape[-1] != self.dim:
            raise ANNIndexError(
                f"dimension mismatch: expected dim={self.dim}, got {vector.shape[-1]}"
            )

        if self.index.ntotal == 0:
            return [], []

        vec32 = self._to_float32(np.asarray(vector))
        vec1d = vec32.flatten()
        vec_bytes = (
            vec1d.tobytes()
            if hasattr(vec1d, "tobytes")
            else struct.pack(f"{len(vec1d)}f", *[float(x) for x in vec1d])
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

        start = perf_counter_ns()
        try:
            with self._lock.reader_lock():
                distances, int_ids = self.index.search(vec, k)
        except RuntimeError as exc:
            _QUERY_ERR.inc()
            raise ANNIndexError("FAISS search failed") from exc

        if int_ids.size == 0:
            self._cache[key] = ([], [])
            return [], []
        latency_sec = (perf_counter_ns() - start) / 1e9
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

    def get_vector(self, id: str) -> NDArray[Any] | None:
        """Return stored vector for ``id`` if present."""
        int_id = self._reverse_id_map.get(id)
        if int_id is None:
            return None
        vec = self._vectors.get(int_id)
        if vec is None:
            return None
        np = require_numpy()
        return cast("NDArray[Any]", np.asarray(vec, dtype=np.float32).copy())

    # ─────────────────────── Rebuild / IO ────────────────────────
    def rebuild(self, vectors: NDArray[Any], ids: Sequence[str]) -> bool:
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
                self._stats.last_rebuild = perf_counter_ns() / 1e9
                self._cache.clear()
                log.info("Index rebuilt with %d vectors", len(ids))
            return True
        except (RuntimeError, ValueError) as e:
            log.error(f"Index rebuild failed: {e}")
            return False

    def _rotate_key_async(self) -> None:
        """Trigger asynchronous key rotation if encryption is enabled."""
        if not self._crypto:
            return
        try:
            loop = get_loop()
            loop.create_task(self._crypto.maybe_rotate_keys())
        except RuntimeError:
            asyncio.run(self._crypto.maybe_rotate_keys())

    def save(self, path: str) -> None:
        """Save index and ID map to disk. Encrypts when configured."""
        try:
            with self._lock.writer_lock():
                data = faiss.serialize_index(self.index)
                map_data = json.dumps(self._id_map).encode()
                if self._crypto:
                    self._rotate_key_async()
                    data = self._crypto.encrypt(data).encode()
                    map_data = self._crypto.encrypt(map_data).encode()
                    Path(path).write_bytes(data)
                    Path(path).with_suffix(".map.json").write_bytes(map_data)
                else:
                    Path(path).write_bytes(data)
                    Path(path).with_suffix(".map.json").write_text(json.dumps(self._id_map))
                log.info("Index saved to %s", path)
        except (OSError, RuntimeError, ValueError) as e:
            log.error(f"Failed to save index: {e}")

    def load(self, path: str) -> None:
        """Load index and ID map from disk. Decrypts when configured."""
        try:
            with self._lock.writer_lock():
                raw = Path(path).read_bytes()
                if self._crypto:
                    raw = self._crypto.decrypt(raw.decode())
                self.index = faiss.deserialize_index(raw)
                map_path = Path(path).with_suffix(".map.json")
                if map_path.exists():
                    mraw = map_path.read_bytes()
                    if self._crypto:
                        mraw = self._crypto.decrypt(mraw.decode())
                        self._id_map = {int(k): v for k, v in json.loads(mraw).items()}
                    else:
                        self._id_map = {int(k): v for k, v in json.loads(mraw.decode()).items()}
                    self._reverse_id_map = {v: int(k) for k, v in self._id_map.items()}
                self._stats.total_vectors = self.index.ntotal
                _INDEX_SIZE.set(self._stats.total_vectors)
                self._cache.clear()
                log.info("Index loaded from %s", path)
        except (OSError, RuntimeError, ValueError) as e:
            log.error(f"Failed to load index: {e}")

    # ─────────────────────── Info ────────────────────────
    def stats(self) -> IndexStats:
        """Return current index statistics."""
        return self._stats


class MultiModalFaissIndex:
    """
    Manage separate FAISS indices for multiple modalities.

    All keyword arguments are forwarded to :class:`FaissHNSWIndex`, enabling
    access to any underlying FAISS option such as IVF, PQ, GPU, etc. Individual
    modalities may override any of these parameters by passing a mapping of
    options instead of just the dimension.
    """

    def __init__(
        self,
        vector_dims: dict[str, int | dict[str, Any]] | None = None,
        **faiss_kwargs: Any,
    ) -> None:
        self._indices: dict[str, FaissHNSWIndex] = {}
        self._config: dict[str, dict[str, Any]] = {}
        if vector_dims:
            for mod, cfg in vector_dims.items():
                if isinstance(cfg, dict):
                    dim = cfg.get("dim")
                    if dim is None:
                        raise ValueError(f"missing 'dim' for modality {mod}")
                    params = {k: v for k, v in cfg.items() if k != "dim"}
                else:
                    dim = int(cfg)
                    params = {}
                params = {**faiss_kwargs, **params}
                self._indices[mod] = FaissHNSWIndex(dim=dim, **params)
                self._config[mod] = {"dim": dim, **params}

    def _get_index(self, modality: str) -> FaissHNSWIndex:
        """Retrieve the index for ``modality`` or raise ``KeyError``."""
        try:
            return self._indices[modality]
        except KeyError as exc:  # pragma: no cover - simple error path
            raise KeyError(
                f"unknown modality '{modality}'; available: {list(self._indices)}"
            ) from exc

    # Basic routing operations -------------------------------------------------
    def add_vectors(
        self, modality: str, ids: list[str], vectors: NDArray[Any], **kwargs: Any
    ) -> None:
        self._get_index(modality).add_vectors(ids, vectors, **kwargs)

    def search(
        self,
        modality: str,
        vector: NDArray[Any],
        *,
        k: int = 5,
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        return self._get_index(modality).search(vector, k=k, ef_search=ef_search)

    def auto_tune(self, samples: dict[str, NDArray[Any]]) -> dict[str, tuple[int, int, int]]:
        """
        Run :meth:`FaissHNSWIndex.auto_tune` for each modality.

        Parameters
        ----------
        samples:
            Mapping of modality name to sample vectors for that modality.

        Returns
        -------
        dict[str, tuple[int, int, int]]
            Mapping of modality to ``(M, ef_construction, ef_search)`` tuples.

        """
        results: dict[str, tuple[int, int, int]] = {}
        for mod, vecs in samples.items():
            results[mod] = self._get_index(mod).auto_tune(vecs)
        return results

    # Persistence --------------------------------------------------------------
    def save(self, path: str) -> None:
        base = Path(path)
        meta: dict[str, dict[str, Any]] = {}
        for mod, idx in self._indices.items():
            idx.save(str(base.with_suffix(base.suffix + f".{mod}")))
            meta[mod] = self._config.get(mod, {"dim": idx.stats().dim})
        meta_path = base.with_suffix(base.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta))

    def load(self, path: str) -> None:
        base = Path(path)
        meta_path = base.with_suffix(base.suffix + ".meta.json")
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            self._indices = {}
            self._config = {}
            for mod, cfg in metadata.items():
                cfg = dict(cfg)
                dim = int(cfg.pop("dim"))
                params = cfg
                self._indices[mod] = FaissHNSWIndex(dim=dim, **params)
                self._config[mod] = {"dim": dim, **params}
                p = base.with_suffix(base.suffix + f".{mod}")
                if p.exists():
                    self._indices[mod].load(str(p))
        else:
            for mod, idx in self._indices.items():
                p = base.with_suffix(base.suffix + f".{mod}")
                if p.exists():
                    idx.load(str(p))

    # Stats -------------------------------------------------------------------
    def stats(self, modality: str | None = None) -> IndexStats:
        if modality is not None:
            return self._get_index(modality).stats()
        total = sum(idx.stats().total_vectors for idx in self._indices.values())
        dim = next(iter(self._indices.values())).stats().dim if self._indices else 0
        return IndexStats(dim=dim, total_vectors=total)
