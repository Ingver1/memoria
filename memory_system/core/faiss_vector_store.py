from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from memory_system.config.settings import UnifiedSettings
from memory_system.core.index import FaissHNSWIndex, MultiModalFaissIndex
from memory_system.core.interfaces import VectorStore


class FaissVectorStore(VectorStore):
    """VectorStore implementation backed by FAISS indices."""

    def __init__(self, settings: UnifiedSettings) -> None:
        self.settings = settings
        vector_dims = getattr(settings.model, "vector_dims", None)
        if vector_dims:
            self._multimodal = True
            self._index = MultiModalFaissIndex(
                vector_dims,
                M=settings.model.hnsw_m,
                ef_construction=settings.model.hnsw_ef_construction,
                ef_search=settings.model.hnsw_ef_search,
                index_type=settings.model.index_type,
                use_gpu=settings.model.use_gpu,
                ivf_nlist=settings.model.ivf_nlist,
                ivf_nprobe=settings.model.ivf_nprobe,
                pq_m=settings.model.pq_m,
                pq_bits=settings.model.pq_bits,
            )
        else:
            self._multimodal = False
            self._index = FaissHNSWIndex(
                dim=settings.model.vector_dim,
                M=settings.model.hnsw_m,
                ef_construction=settings.model.hnsw_ef_construction,
                ef_search=settings.model.hnsw_ef_search,
                index_type=settings.model.index_type,
                use_gpu=settings.model.use_gpu,
                ivf_nlist=settings.model.ivf_nlist,
                ivf_nprobe=settings.model.ivf_nprobe,
                pq_m=settings.model.pq_m,
                pq_bits=settings.model.pq_bits,
            )
        self._path = settings.database.vec_path
        if self._path.exists():
            self._index.load(str(self._path))

    # ------------------------------------------------------------------
    @property
    def index(self) -> FaissHNSWIndex | MultiModalFaissIndex:
        return self._index

    # ------------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text") -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if self._multimodal:
            self._index.add_vectors(modality, list(ids), arr)
        else:
            self._index.add_vectors(list(ids), arr)

    def search(
        self,
        vector: NDArray[np.float32],
        *,
        k: int = 5,
        modality: str = "text",
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        vec = np.asarray(vector, dtype=np.float32)
        if self._multimodal:
            return self._index.search(modality, vec, k=k, ef_search=ef_search)
        return self._index.search(vec, k=k, ef_search=ef_search)

    def update(self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text") -> None:
        self.delete(ids, modality=modality)
        self.add(ids, vectors, modality=modality)

    def delete(self, ids: Sequence[str], *, modality: str = "text") -> None:
        if self._multimodal:
            self._index._indices[modality].remove_ids(ids)  # type: ignore[attr-defined]
        else:
            self._index.remove_ids(ids)

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self._path
        self._index.save(str(p))

    def load(self, path: str | None = None) -> None:
        p = Path(path) if path else self._path
        if p.exists():
            self._index.load(str(p))

    def stats(self, modality: str | None = None):
        if self._multimodal:
            return self._index.stats(modality)
        return self._index.stats()

    @property
    def ef_search(self) -> int:
        if self._multimodal:
            first = next(iter(self._index._indices.values()))  # type: ignore[attr-defined]
            return first.ef_search
        return self._index.ef_search
