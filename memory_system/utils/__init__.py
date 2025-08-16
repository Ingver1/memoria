"""Utilities for Unified Memory System (metrics, cache, security, etc.)."""

# Avoid importing heavy optional dependencies at module import time.  Everything
# is loaded lazily via ``__getattr__`` to keep the package importable in
# minimal environments (e.g. documentation builds) where ``numpy`` and other
# extras might be absent.

from typing import Any

__all__ = ["NeighborCache", "get_loop", "knn_lm_mix", "require_numpy"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial forwarding
    if name in {"NeighborCache", "knn_lm_mix"}:
        from .knn_lm import NeighborCache, knn_lm_mix

        return {"NeighborCache": NeighborCache, "knn_lm_mix": knn_lm_mix}[name]
    if name == "get_loop":
        from .loop import get_loop

        return get_loop
    if name == "require_numpy":
        from .numpy import require_numpy

        return require_numpy
    raise AttributeError(f"module {__name__} has no attribute {name}")
