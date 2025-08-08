"""Local embedding helper using SentenceTransformer.

Provides ``embed`` function compatible with ``UnifiedSettings`` integration.
"""

from __future__ import annotations

import logging

import numpy as np

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str | list[str]) -> np.ndarray:
    """Return embeddings for ``text`` without external API calls."""
    if isinstance(text, str):
        text = [text]

    log = logging.getLogger(__name__)
    log.info("Embedding batch size %d", len(text))

    if not any(text):
        model_dim = _model.get_sentence_embedding_dimension()
        if len(text) == 1:
            return np.zeros(model_dim, dtype=np.float32)
        raise ValueError("text cannot be empty")
        
    vecs = _model.encode(text)
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    if len(vecs) == 1:
        return vecs[0]
    return vecs
