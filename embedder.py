"""Local embedding helper using SentenceTransformer.

Provides ``embed`` function compatible with ``UnifiedSettings`` integration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

try:  # optional numpy
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:  # optional sentence-transformers
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as _np

_model: SentenceTransformer | None = None


def _require_numpy() -> Any:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for local embeddings. Install ai-memory[embedder]."
        )
    return np


def _require_sentence_transformers() -> Any:
    if SentenceTransformer is None:
        raise ModuleNotFoundError(
            "sentence-transformers is required for local embeddings."
        )
    return SentenceTransformer


def _get_model() -> Any:
    global _model
    if _model is None:
        st = _require_sentence_transformers()
        _model = st("all-MiniLM-L6-v2")
    return _model


def embed(text: str | list[str]) -> "_np.ndarray":
    """Return embeddings for ``text`` without external API calls."""
    np = _require_numpy()
    model = _get_model()
    if isinstance(text, str):
        text = [text]

    log = logging.getLogger(__name__)
    log.info("Embedding batch size %d", len(text))

    if not any(text):
        model_dim = model.get_sentence_embedding_dimension()
        if len(text) == 1:
            return np.zeros(model_dim, dtype=np.float32)
        raise ValueError("text cannot be empty")

    vecs = model.encode(text)
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    if len(vecs) == 1:
        return vecs[0]
    return vecs
