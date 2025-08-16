"""Lazy import helpers for optional dependencies."""

# ruff: noqa: ANN401, PLW0603, TRY003, EM101

from __future__ import annotations

from importlib import import_module
from threading import Lock
from typing import Any

_faiss: Any | None = None
_httpx: Any | None = None
_np: Any | None = None
_st: Any | None = None

_faiss_lock = Lock()
_httpx_lock = Lock()
_np_lock = Lock()
_st_lock = Lock()


def require_faiss() -> Any:
    """Return :mod:`faiss` or raise a helpful error if missing."""
    global _faiss
    if _faiss is None:
        with _faiss_lock:
            if _faiss is None:
                try:  # pragma: no cover - optional dependency
                    _faiss = import_module("faiss")
                except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
                    raise ModuleNotFoundError(
                        "faiss is required for vector search. Install ai-memory[faiss]."
                    ) from exc
    return _faiss


def require_httpx() -> Any:
    """Return :mod:`httpx` or raise a helpful error if missing."""
    global _httpx
    if _httpx is None:
        with _httpx_lock:
            if _httpx is None:
                try:  # pragma: no cover - optional dependency
                    _httpx = import_module("httpx")
                except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
                    raise ModuleNotFoundError(
                        "httpx is required for HTTP features. Install ai-memory[cli] or "
                        "ai-memory[api].",
                    ) from exc
    return _httpx


def require_numpy() -> Any:
    """Return :mod:`numpy` or raise a helpful error if missing."""
    global _np
    if _np is None:
        with _np_lock:
            if _np is None:
                try:  # pragma: no cover - optional dependency
                    _np = import_module("numpy")
                except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
                    raise ModuleNotFoundError(
                        "numpy is required for this operation. Install with "
                        "'pip install ai-memory[numpy]' or 'pip install ai-memory[full]'.",
                    ) from exc
    return _np


def require_sentence_transformers() -> Any:
    """Return :class:`SentenceTransformer` or raise if it's unavailable."""
    global _st
    if _st is None:
        with _st_lock:
            if _st is None:
                try:  # pragma: no cover - optional dependency
                    module = import_module("memory_system._vendor.sentence_transformers")
                    _st = module.SentenceTransformer
                except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
                    raise ModuleNotFoundError(
                        "sentence-transformers is required for embedding features. Install with "
                        "'pip install sentence-transformers' or 'pip install ai-memory[full]'.",
                    ) from exc
    return _st


__all__ = [
    "require_faiss",
    "require_httpx",
    "require_numpy",
    "require_sentence_transformers",
]
