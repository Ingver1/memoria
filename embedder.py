from __future__ import annotations

"""Local embedding helper using SentenceTransformer.

Provides ``embed`` function compatible with ``UnifiedSettings`` integration.
"""


import numpy as np

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str | list[str]) -> np.ndarray:
    """Return embeddings for ``text`` without external API calls."""
    if isinstance(text, str):
        text = [text]
    vecs = _model.encode(text)
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Ensure norms is always an ndarray for safe normalization
    if isinstance(norms, float):
        norms_safe = np.array([norms if norms != 0 else 1], dtype=np.float32)
        vecs = vecs / norms_safe[0]
    else:
        norms_safe = np.array(norms, dtype=np.float32).flatten()
        for i in range(norms_safe.shape[0]):
            if norms_safe[i] == 0:
                norms_safe[i] = 1
        # Element-wise division for each vector
        for i in range(vecs.shape[0]):
            vecs[i] = vecs[i] / norms_safe[i]
    if len(vecs) == 1:
        return np.asarray(vecs[0], dtype=np.float32)
    return vecs
