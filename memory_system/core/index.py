from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from ._faiss_mod import (
        ANNIndexError,
        FaissHNSWIndex,
        IndexStats,
        MultiModalFaissIndex,
        _INDEX_SIZE,
        _QUERY_CNT,
        _QUERY_ERR,
        _QUERY_LATENCY,
        _VEC_ADDED,
        _VEC_DELETED,
    )
else:
    ANNIndexError = FaissHNSWIndex = IndexStats = MultiModalFaissIndex = Any  # type: ignore[assignment]

from . import _faiss_mod

ANNIndexError = _faiss_mod.ANNIndexError
FaissHNSWIndex = _faiss_mod.FaissHNSWIndex
IndexStats = _faiss_mod.IndexStats
MultiModalFaissIndex = _faiss_mod.MultiModalFaissIndex
_INDEX_SIZE = _faiss_mod._INDEX_SIZE
_QUERY_CNT = _faiss_mod._QUERY_CNT
_QUERY_ERR = _faiss_mod._QUERY_ERR
_QUERY_LATENCY = _faiss_mod._QUERY_LATENCY
_VEC_ADDED = _faiss_mod._VEC_ADDED
_VEC_DELETED = _faiss_mod._VEC_DELETED

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
