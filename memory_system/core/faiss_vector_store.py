from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from memory_system.core.index import (
    FaissHNSWIndex,
    IndexStats,
    MultiModalFaissIndex,
)
from memory_system.core.interfaces import VectorStore
from memory_system.settings import UnifiedSettings
from memory_system.utils.dependencies import require_faiss
from memory_system.utils.loop import get_loop
from memory_system.utils.metrics import LAT_SEARCH, MET_ERRORS_TOTAL

log = logging.getLogger(__name__)

# Fallback when the faiss stub lacks full functionality
if TYPE_CHECKING:
    import faiss
else:
    faiss = require_faiss()
_FAISS_OK = all(
    hasattr(faiss, attr)
    for attr in ["IndexHNSWFlat", "IndexIDMap2", "METRIC_L2", "METRIC_INNER_PRODUCT"]
)


class _ListIndex:
    """Minimal in-memory index used when FAISS is unavailable."""

    def __init__(self, dim: int, space: str = "cosine") -> None:
        self.dim = dim
        self.space = space
        self.ef_search = 0
        self._vectors: dict[str, list[float]] = {}

    # FAISS-compatible methods -------------------------------------------------
    def add_vectors(self, ids: list[str], vectors: NDArray[np.float32]) -> None:
        if self.space == "cosine":
            arr = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
            vec_list = arr.tolist()
        else:
            vec_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        for mid, vec in zip(ids, vec_list, strict=False):
            self._vectors[mid] = list(vec)

    def search(
        self, vec: NDArray[np.float32], k: int = 5, ef_search: int | None = None
    ) -> tuple[list[str], list[float]]:
        if self.space == "cosine":
            arr = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm != 0:
                arr = arr / norm
            q = arr.tolist()
        else:
            q = vec.tolist() if hasattr(vec, "tolist") else vec
        sims: list[tuple[float, str]] = []
        for mid, v in self._vectors.items():
            score = float(sum(a * b for a, b in zip(q, v, strict=False)))
            sims.append((score, mid))
        sims.sort(key=lambda x: x[0], reverse=True)
        ids = [mid for _, mid in sims[:k]]
        if self.space == "cosine":
            # Convert cosine similarity into a distance measure comparable to
            # FAISS' inner‑product indices where higher similarity implies a
            # lower distance.  Returning ``1 - score`` mirrors the behaviour of
            # real cosine distance while keeping the fallback lightweight.
            dists = [1.0 - score for score, _ in sims[:k]]
        else:
            # For non‑cosine spaces fall back to a basic Euclidean distance.
            dists = [
                sum((a - b) ** 2 for a, b in zip(q, self._vectors[mid], strict=False)) ** 0.5
                for _, mid in sims[:k]
            ]
        return ids, dists

    def set_ef_search(self, value: int) -> None:
        """Update the ef_search hint for this fallback index."""
        self.ef_search = int(value)

    def remove_ids(self, ids: Sequence[str]) -> None:
        for mid in ids:
            self._vectors.pop(mid, None)

    def rebuild(self, vectors: NDArray[np.float32], ids: list[str]) -> None:
        vec_list = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        self._vectors = {mid: list(vec) for mid, vec in zip(ids, vec_list, strict=False)}

    def save(self, path: str) -> None:  # pragma: no cover - no-op
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._vectors, fh)
        except Exception:
            # Persistence is a best-effort cache; ignore failures to keep the
            # fallback lightweight and avoid introducing mandatory file system
            # permissions for tests.
            return

    def load(self, path: str) -> None:  # pragma: no cover - no-op
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._vectors = {str(k): list(map(float, v)) for k, v in data.items()}
        except Exception:
            # Corrupted or unreadable cache; start with an empty index.
            self._vectors = {}

    def stats(self) -> IndexStats:
        return IndexStats(dim=self.dim, total_vectors=len(self._vectors))


class FaissVectorStore(VectorStore):
    """VectorStore implementation backed by FAISS indices."""

    def __init__(self, settings: UnifiedSettings) -> None:
        self.settings = settings
        vector_dims = getattr(settings.model, "vector_dims", None)
        fcfg = settings.faiss
        dataset_size = fcfg.dataset_size
        params = {
            "M": fcfg.M,
            "ef_construction": fcfg.ef_construction,
            "ef_search": fcfg.ef_search,
            "ivf_nlist": fcfg.nlist,
            "ivf_nprobe": fcfg.nprobe,
            "pq_m": fcfg.pq_m,
            "pq_bits": fcfg.pq_bits,
        }
        if dataset_size:
            if params["ivf_nlist"] is None:
                params["ivf_nlist"] = max(1, int(fcfg.nlist_scale * math.sqrt(dataset_size or 0)))
            if params["ef_search"] is None:
                ivf_nlist = params["ivf_nlist"] or 0
                params["ef_search"] = max(1, int(ivf_nlist * fcfg.ef_search_scale))
            if params["ef_construction"] is None:
                ef_search_val = params["ef_search"] or 0
                params["ef_construction"] = max(
                    ef_search_val,
                    int(ef_search_val * fcfg.ef_construction_scale),
                )
            if params["pq_m"] is None:
                params["pq_m"] = max(1, settings.model.vector_dim // fcfg.pq_m_div)
            if params["pq_bits"] is None:
                params["pq_bits"] = 8
        if params["ef_construction"] is None:
            params["ef_construction"] = FaissHNSWIndex.DEFAULT_EF_CONSTRUCTION
        if params["ef_search"] is None:
            params["ef_search"] = FaissHNSWIndex.DEFAULT_EF_SEARCH
        if params["ivf_nlist"] is None:
            params["ivf_nlist"] = FaissHNSWIndex.DEFAULT_IVF_NLIST
        if params["ivf_nprobe"] is None:
            params["ivf_nprobe"] = FaissHNSWIndex.DEFAULT_IVF_NPROBE
        if params["pq_m"] is None:
            params["pq_m"] = FaissHNSWIndex.DEFAULT_PQ_M
        if params["pq_bits"] is None:
            params["pq_bits"] = FaissHNSWIndex.DEFAULT_PQ_BITS
        self._autotune = bool(fcfg.autotune)
        index_type = fcfg.index_type
        use_gpu = fcfg.use_gpu

        self._index: FaissHNSWIndex | MultiModalFaissIndex | _ListIndex
        if not _FAISS_OK:
            self._multimodal = False
            self._index = _ListIndex(
                settings.model.vector_dim,
                getattr(settings.model, "space", "cosine"),
            )
        elif vector_dims:
            self._multimodal = True
            self._index = MultiModalFaissIndex(
                vector_dims,
                index_type=index_type,
                use_gpu=use_gpu,
                M=params["M"],
                ef_construction=params["ef_construction"],
                ef_search=params["ef_search"],
                ivf_nlist=params["ivf_nlist"],
                ivf_nprobe=params["ivf_nprobe"],
                pq_m=params["pq_m"],
                pq_bits=params["pq_bits"],
            )
        else:
            self._multimodal = False
            self._index = FaissHNSWIndex(
                dim=settings.model.vector_dim,
                index_type=index_type,
                use_gpu=use_gpu,
                M=params["M"],
                ef_construction=params["ef_construction"],
                ef_search=params["ef_search"],
                ivf_nlist=params["ivf_nlist"],
                ivf_nprobe=params["ivf_nprobe"],
                pq_m=params["pq_m"],
                pq_bits=params["pq_bits"],
            )
        log.info(
            "FAISS parameters: M=%d efC=%d efS=%d nlist=%d nprobe=%d pq_m=%d pq_bits=%d",
            params["M"],
            params["ef_construction"],
            params["ef_search"],
            params["ivf_nlist"],
            params["ivf_nprobe"],
            params["pq_m"],
            params["pq_bits"],
        )
        self._path = settings.database.vec_path
        if self._path.exists():
            self._index.load(str(self._path))

        if self._autotune and _FAISS_OK:
            self._run_autotune_on_init()

        # Asynchronous save management
        self._loop: asyncio.AbstractEventLoop | None = None
        try:
            self._loop = get_loop()
        except RuntimeError:
            self._loop = None
        self._save_lock = asyncio.Lock()
        self._save_task: asyncio.Task[None] | None = None
        self._last_save = time.monotonic()
        self._pending_changes = 0
        self._save_time_threshold = 5.0
        self._save_size_threshold = max(1, getattr(settings.model, "batch_add_size", 100))

    # ------------------------------------------------------------------
    @property
    def index(self) -> FaissHNSWIndex | MultiModalFaissIndex | _ListIndex:
        return self._index

    def _schedule_save(self, count: int) -> None:
        self._pending_changes += count
        now = time.monotonic()
        if (
            now - self._last_save < self._save_time_threshold
            and self._pending_changes < self._save_size_threshold
        ):
            return
        self._pending_changes = 0
        self._last_save = now
        loop = self._loop
        if loop is None:
            try:
                loop = get_loop()
            except RuntimeError:
                return
            self._loop = loop

        def _launch() -> None:
            if self._save_task is None or self._save_task.done():
                self._save_task = loop.create_task(self._async_save())

        loop.call_soon_threadsafe(_launch)

    async def _async_save(self) -> None:
        async with self._save_lock:
            await asyncio.to_thread(self.save)

    def _run_autotune_on_init(self) -> None:
        if self._multimodal:
            idx = cast("MultiModalFaissIndex", self._index)
            samples: dict[str, NDArray[np.float32]] = {}
            for mod, sub in idx._indices.items():
                ids = list(sub._id_map.keys())
                if not ids:
                    continue
                k = min(len(ids), 100)
                sample_ids = random.sample(ids, k)
                vecs = np.vstack([sub.index.reconstruct(int(i)) for i in sample_ids]).astype(
                    np.float32
                )
                samples[mod] = vecs
            if not samples:
                return
            results = idx.auto_tune(samples)
            first = next(iter(results.values()))
            for mod, (M, ef_c, ef_s) in results.items():
                base = faiss.downcast_index(idx._indices[mod].index.index)
                if hasattr(base, "hnsw"):
                    base.hnsw.efConstruction = ef_c
                    base.hnsw.efSearch = ef_s
                idx._indices[mod].ef_search = ef_s
            try:
                object.__setattr__(self.settings.faiss, "M", first[0])
                object.__setattr__(self.settings.faiss, "ef_construction", first[1])
                object.__setattr__(self.settings.faiss, "ef_search", first[2])
                object.__setattr__(self.settings.faiss, "autotune", False)
            except AttributeError:
                pass
            self._autotune = False
        else:
            index = cast("FaissHNSWIndex", self._index)
            ids = list(index._id_map.keys())
            if not ids:
                return
            k = min(len(ids), 100)
            sample_ids = random.sample(ids, k)
            vecs = np.vstack([index.index.reconstruct(int(i)) for i in sample_ids]).astype(
                np.float32
            )
            M, ef_c, ef_s = index.auto_tune(vecs)
            base = faiss.downcast_index(index.index.index)
            if hasattr(base, "hnsw"):
                base.hnsw.efConstruction = ef_c
                base.hnsw.efSearch = ef_s
            index.ef_search = ef_s
            try:
                object.__setattr__(self.settings.faiss, "M", M)
                object.__setattr__(self.settings.faiss, "ef_construction", ef_c)
                object.__setattr__(self.settings.faiss, "ef_search", ef_s)
                object.__setattr__(self.settings.faiss, "autotune", False)
            except AttributeError:
                pass
            self._autotune = False

    # ------------------------------------------------------------------
    def add(
        self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text"
    ) -> None:
        try:
            if not _FAISS_OK:
                cast("_ListIndex", self._index).add_vectors(list(ids), vectors)
                return
            arr = np.asarray(vectors, dtype=np.float32)
            if self._autotune and not self._multimodal:
                index = cast("FaissHNSWIndex", self._index)
                if getattr(index.index, "ntotal", 0) == 0:
                    M, ef_c, ef_s = index.auto_tune(arr)
                    base = faiss.downcast_index(index.index.index)
                    if hasattr(base, "hnsw"):
                        base.hnsw.efConstruction = ef_c
                        base.hnsw.efSearch = ef_s
                    index.ef_search = ef_s
                    try:
                        object.__setattr__(self.settings.faiss, "M", M)
                        object.__setattr__(self.settings.faiss, "ef_construction", ef_c)
                        object.__setattr__(self.settings.faiss, "ef_search", ef_s)
                        object.__setattr__(self.settings.faiss, "autotune", False)
                    except AttributeError:
                        pass
                    self._autotune = False
            if self._multimodal:
                cast("MultiModalFaissIndex", self._index).add_vectors(modality, list(ids), arr)
            else:
                cast("FaissHNSWIndex", self._index).add_vectors(list(ids), arr)
            self._schedule_save(len(ids))
        except Exception as exc:  # pragma: no cover - defensive
            MET_ERRORS_TOTAL.labels(type="vector_store", component="add").inc()
            log.error("Faiss add failed: %s", exc)
            raise

    def search(
        self,
        vector: NDArray[np.float32],
        *,
        k: int = 5,
        modality: str = "text",
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        try:
            with LAT_SEARCH.time():
                if not _FAISS_OK:
                    return cast("_ListIndex", self._index).search(vector, k=k, ef_search=ef_search)
                vec = np.asarray(vector, dtype=np.float32)
                if self._multimodal:
                    return cast("MultiModalFaissIndex", self._index).search(
                        modality, vec, k=k, ef_search=ef_search
                    )
                return cast("FaissHNSWIndex", self._index).search(vec, k=k, ef_search=ef_search)
        except Exception as exc:  # pragma: no cover - defensive
            MET_ERRORS_TOTAL.labels(type="vector_store", component="search").inc()
            log.error("Faiss search failed: %s", exc)
            raise

    def update(
        self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text"
    ) -> None:
        self.delete(ids, modality=modality)
        self.add(ids, vectors, modality=modality)

    def delete(self, ids: Sequence[str], *, modality: str = "text") -> None:
        try:
            if self._multimodal:
                index = cast("MultiModalFaissIndex", self._index)
                try:
                    index._get_index(modality).remove_ids(ids)
                except KeyError as exc:
                    raise ValueError(f"unknown modality '{modality}'") from exc
            else:
                cast("FaissHNSWIndex", self._index).remove_ids(ids)
            self._schedule_save(len(ids))
        except Exception as exc:  # pragma: no cover - defensive
            MET_ERRORS_TOTAL.labels(type="vector_store", component="delete").inc()
            log.error("Faiss delete failed: %s", exc)
            raise

    def rebuild(self, modality: str, vectors: NDArray[np.float32], ids: Sequence[str]) -> None:
        try:
            if not _FAISS_OK:
                cast("_ListIndex", self._index).rebuild(vectors, list(ids))
                self._schedule_save(len(ids))
                return
            arr = np.asarray(vectors, dtype=np.float32)
            if self._multimodal:
                index = cast("MultiModalFaissIndex", self._index)
                try:
                    index._get_index(modality).rebuild(arr, list(ids))
                except KeyError as exc:
                    raise ValueError(f"unknown modality '{modality}'") from exc
            else:
                cast("FaissHNSWIndex", self._index).rebuild(arr, list(ids))
            self._schedule_save(len(ids))
        except Exception as exc:  # pragma: no cover - defensive
            MET_ERRORS_TOTAL.labels(type="vector_store", component="rebuild").inc()
            log.error("Faiss rebuild failed: %s", exc)
            raise

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self._path
        self._index.save(str(p))

    def load(self, path: str | None = None) -> None:
        p = Path(path) if path else self._path
        if p.exists():
            self._index.load(str(p))

    def stats(self, modality: str | None = None) -> dict[str, Any]:
        if self._multimodal:
            return cast("dict[str, Any]", cast("MultiModalFaissIndex", self._index).stats(modality))
        return cast("dict[str, Any]", cast("FaissHNSWIndex", self._index).stats())

    @property
    def ef_search(self) -> int:
        if self._multimodal:
            index = cast("MultiModalFaissIndex", self._index)
            first = next(iter(index._indices.values()))
            return first.ef_search
        return cast("FaissHNSWIndex", self._index).ef_search

    def set_ef_search(self, value: int) -> None:
        """Update the HNSW ``efSearch`` parameter if supported."""
        if not _FAISS_OK:
            cast("_ListIndex", self._index).set_ef_search(value)
            return
        if self._multimodal:
            mm = cast("MultiModalFaissIndex", self._index)
            for idx in mm._indices.values():
                base = faiss.downcast_index(idx.index.index)
                if hasattr(base, "hnsw"):
                    base.hnsw.efSearch = int(value)
                    idx.ef_search = int(value)
            return
        idx = cast("FaissHNSWIndex", self._index)
        base = faiss.downcast_index(idx.index.index)
        if hasattr(base, "hnsw"):
            base.hnsw.efSearch = int(value)
            idx.ef_search = int(value)

    async def close(self) -> None:
        """Persist the FAISS index to disk asynchronously."""
        if self._save_task is not None:
            await self._save_task
        await asyncio.to_thread(self.save)
