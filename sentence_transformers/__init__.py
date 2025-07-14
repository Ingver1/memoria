import hashlib
from typing import cast

import numpy as np


class SentenceTransformer:
    """Minimal stub of :class:`SentenceTransformer` for offline tests."""

    def __init__(self, model_name: str) -> None:
        if model_name != "all-MiniLM-L6-v2":
            raise ValueError(f"model {model_name} not found")
        self.model_name = model_name
        self._dim = 384

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Return deterministic embeddings for *texts*."""
        if isinstance(texts, str):
            texts = [texts]
        vectors: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # repeat or truncate to self._dim
            if arr.size < self._dim:
                reps = (self._dim + arr.size - 1) // arr.size
                arr = np.array(np.tile(arr, reps)[: self._dim])
            else:
                arr = np.array(arr[: self._dim])
            norm_val = float(cast(float, np.linalg.norm(arr)))
            if norm_val != 0:
                arr = arr / norm_val
            vectors.append(list(arr))
        return np.array(vectors, dtype=np.float32)
