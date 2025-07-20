from __future__ import annotations

"""Local embedding helper using SentenceTransformer.

Provides ``embed`` function compatible with ``UnifiedSettings`` integration.
"""

import inspect

import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str | list[str]) -> np.ndarray:
    """Return embeddings for ``text`` without external API calls."""
    if isinstance(text, str):
        text = [text]
    if "normalize_embeddings" in inspect.signature(_model.encode).parameters:
        vecs = _model.encode(text, normalize_embeddings=True)
    else:
        vecs = _model.encode(text)
        vecs = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms == 0, 1, norms)
    return vecs[0] if len(vecs) == 1 else vecs
