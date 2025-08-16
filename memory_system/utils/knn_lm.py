"""
kNN-LM mixing utilities.

This module provides helper primitives for mixing language model
probabilities with probabilities derived from a k-nearest neighbour search
over a retrieval index.  A small in-memory cache is maintained to avoid
repeated searches against the global index.
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
from logging import getLogger
from math import exp
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol, cast

from .dependencies import require_numpy

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Iterable, Sequence

    import numpy as np
    from numpy import ndarray
else:
    np = require_numpy()
    ndarray = Any


class _IndexProtocol(Protocol):
    def search(self, query: Sequence[float], k: int) -> Iterable[tuple[int, float]]: ...


__all__ = ["NeighborCache", "knn_lm_mix"]


logger = getLogger(__name__)

_STATS = {
    "calls": 0,
    "lambda_sum": 0.0,
    "knn_better": 0,
    "total_tokens": 0,
}


def knn_lm_mix(
    probs_lm: ndarray,
    probs_knn: ndarray | None,
    lambda_: float,
    *,
    max_knn_ratio: float | None = None,
    forbidden_tokens: Sequence[int] | None = None,
) -> ndarray:
    """
    Combine LM and kNN probabilities.

    Parameters
    ----------
    probs_lm:
        Base language model probabilities for the next token.
    probs_knn:
        Probabilities produced from a kNN search.  When ``None`` or all zeros
        the function simply returns ``probs_lm`` - this allows graceful
        fallback when no neighbours are available.
    lambda_:
        Interpolation factor.  ``0`` relies entirely on the LM probabilities
        while ``1`` uses only the kNN distribution.

    """
    if probs_knn is None or not any(probs_knn):
        return probs_lm

    lambda_ = float(max(0.0, min(1.0, lambda_)))

    pairs = list(zip(probs_knn, probs_lm, strict=False))
    better = sum(knn > lm for knn, lm in pairs)
    ratio = better / len(pairs)

    _STATS["calls"] += 1
    _STATS["lambda_sum"] += lambda_
    _STATS["knn_better"] += better
    _STATS["total_tokens"] += len(pairs)
    avg_lambda = _STATS["lambda_sum"] / _STATS["calls"]
    knn_ratio = _STATS["knn_better"] / max(_STATS["total_tokens"], 1)
    logger.debug("kNN>LM ratio: %.4f avg λ: %.4f", knn_ratio, avg_lambda)

    if max_knn_ratio is not None and ratio > max_knn_ratio:
        return probs_lm
    if forbidden_tokens:
        for idx in forbidden_tokens:
            if probs_knn[idx] > probs_lm[idx]:
                return probs_lm

    mixed = [(1.0 - lambda_) * lm + lambda_ * knn for knn, lm in pairs]
    arr = None
    try:  # pragma: no cover - optional numpy
        arr = np.asarray(mixed, dtype=float)
    except Exception:  # pragma: no cover - optional numpy
        arr = None
    return arr if arr is not None else mixed


class NeighborCache:
    """
    Cache kNN probability lookups over a global index.

    The cache stores probability distributions keyed by the raw bytes of the
    query vector.  When a query is not present the index's ``search`` method is
    invoked to retrieve neighbours.  The search method is expected to return an
    iterable of ``(token_id, distance)`` pairs.  Distances are converted into
    probabilities via a softmax over the negative distances.
    """

    def __init__(
        self,
        index: Any,  # noqa: ANN401
        vocab_size: int,
        cache_size: int = 1024,
        ttl: float = 300.0,
    ) -> None:
        """Create a cache around *index*."""
        self._index: _IndexProtocol = cast("_IndexProtocol", index)
        self._vocab = vocab_size
        self._cache: OrderedDict[bytes, tuple[float, ndarray | None]] = OrderedDict()
        self._maxsize = cache_size
        self._ttl = ttl

    def get_probabilities(self, query: Sequence[float], k: int = 1) -> ndarray | None:
        """
        Return kNN probability distribution for ``query``.

        Results are cached based on the raw bytes of ``query``.  When the index
        returns no neighbours ``None`` is cached so subsequent lookups avoid
        repeated searches.
        """
        q = array("f", [float(x) for x in query])
        key = q.tobytes()
        now = monotonic()
        if key in self._cache:
            ts, value = self._cache[key]
            if now - ts < self._ttl:
                self._cache.move_to_end(key)
                return value
            del self._cache[key]

        while self._cache:
            oldest_key, (ts, _) = next(iter(self._cache.items()))
            if now - ts >= self._ttl:
                self._cache.popitem(last=False)
            else:
                break

        neighbours: Iterable[tuple[int, float]] = self._index.search(q, k)
        neighbours = list(neighbours)
        if not neighbours:
            self._cache[key] = (now, None)
            return None

        ids, distances = zip(*neighbours, strict=False)
        weights = [exp(-float(d)) for d in distances]
        total = sum(weights)
        probs = [0.0] * self._vocab
        for token_id, weight in zip(ids, weights, strict=False):
            probs[token_id] += float(weight) / total
        self._cache[key] = (now, probs)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return probs
