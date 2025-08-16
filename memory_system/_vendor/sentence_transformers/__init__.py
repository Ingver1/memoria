from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from numpy import ndarray


class SentenceTransformer:
    """Minimal stub of :class:`SentenceTransformer` for offline tests."""

    _MODELS = {
        "all-MiniLM-L6-v2": 384,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "bge-m3": 1024,
    }

    def __init__(self, model_name: str) -> None:
        if model_name not in self._MODELS:
            raise ValueError(f"model {model_name} not found")
        self.model_name = model_name
        self._dim = self._MODELS[model_name]

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: str | list[str]) -> ndarray:
        """Return deterministic embeddings for *texts*."""
        if isinstance(texts, str):
            texts = [texts]
        vectors: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            # start from deterministic float values derived from the hash
            arr = [b / 255 for b in h]
            # repeat or truncate to match the expected dimension
            if len(arr) < self._dim:
                reps = (self._dim + len(arr) - 1) // len(arr)
                arr = (arr * reps)[: self._dim]
            else:
                arr = arr[: self._dim]
            norm_val = sum(x * x for x in arr) ** 0.5
            if norm_val:
                arr = [x / norm_val for x in arr]
            vectors.append(arr)
        return np.array(vectors, dtype=float)
