from __future__ import annotations

from importlib import import_module
import sys
from typing import Any, cast

try:
    _faiss_mod = cast("Any", import_module("memory_system.core._faiss_hnsw_real"))
    _faiss = sys.modules.get("faiss")
    if _faiss is not None and "tests/_stubs" in str(getattr(_faiss, "__file__", "")):
        raise RuntimeError("FAISS stub detected; using fallback")
except Exception:  # pragma: no cover - fallback
    _faiss_mod = cast("Any", import_module("memory_system.core._faiss_hnsw_fallback"))

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
