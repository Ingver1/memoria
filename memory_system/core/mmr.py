"""Maximal Marginal Relevance utilities."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def mmr(
    items: Sequence[tuple[T, Sequence[float]]],
    query_embed: Sequence[float],
    lambda_: float = 0.7,
    *,
    seed: int | None = 0,
) -> list[T]:
    """Return items ordered by Maximal Marginal Relevance."""
    if not items:
        return []
    rng = random.Random(seed)  # noqa: S311 - deterministic for reproducible tests
    selected: list[tuple[T, Sequence[float]]] = []
    remaining = list(items)
    sims = [_cos(embed, query_embed) for _, embed in remaining]
    while remaining:
        indices = list(range(len(remaining)))
        rng.shuffle(indices)
        if not selected:
            idx = max(indices, key=lambda i: (sims[i], -i))
        else:
            scores: list[float] = []
            for i, (_itm, emb) in enumerate(remaining):
                sim_q = sims[i]
                sim_s = max(_cos(emb, s_emb) for _, s_emb in selected)
                scores.append(lambda_ * sim_q - (1 - lambda_) * sim_s)
            idx = max(indices, key=lambda i: (scores[i], -i))
        selected.append(remaining.pop(idx))
        sims.pop(idx)
    return [item for item, _ in selected]


__all__ = ["mmr"]
