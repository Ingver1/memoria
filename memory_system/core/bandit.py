"""Multi-armed bandit reranking utilities."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from memory_system.unified_memory import Memory


def _ucb_scores(candidates: Sequence[Memory]) -> dict[str, float]:
    total = sum(int((m.metadata or {}).get("trial_count", 0)) for m in candidates) + 1
    scores: dict[str, float] = {}
    for m in candidates:
        meta = m.metadata or {}
        success = int(meta.get("success_count", 0))
        trials = int(meta.get("trial_count", 0))
        mean = success / trials if trials else 0.0
        score = mean + math.sqrt(2 * math.log(total) / (trials + 1))
        scores[m.memory_id] = score
    return scores


def _thompson_scores(candidates: Sequence[Memory], seed: int | None = None) -> dict[str, float]:
    rng = random.Random(seed)  # noqa: S311 - deterministic for reproducible tests
    scores: dict[str, float] = {}
    for m in candidates:
        meta = m.metadata or {}
        success = int(meta.get("success_count", 0))
        trials = int(meta.get("trial_count", 0))
        failures = max(trials - success, 0)
        scores[m.memory_id] = rng.betavariate(success + 1, failures + 1)
    return scores


def rerank_with_bandit(
    candidates: Sequence[Memory],
    *,
    method: str = "ucb",
    weight: float = 1.0,
) -> list[Memory]:
    """
    Rerank ``candidates`` using a multi-armed bandit score.

    Parameters
    ----------
    candidates:
        Result list from :func:`search`.
    method:
        "ucb" or "thompson".
    weight:
        Influence of bandit score relative to original ranking. ``1.0`` means
        bandit score is added directly to the descending positional score.

    """
    if not candidates:
        return []

    if method == "ucb":
        bandit_scores = _ucb_scores(candidates)
    elif method == "thompson":
        bandit_scores = _thompson_scores(candidates, seed=0)
    else:  # pragma: no cover - defensive
        msg = f"Unknown bandit method: {method}"
        raise ValueError(msg)

    n = len(candidates)
    base_scores = {m.memory_id: (n - i) * 1e-6 for i, m in enumerate(candidates)}
    combined = {
        m.memory_id: weight * bandit_scores.get(m.memory_id, 0.0) + base_scores[m.memory_id]
        for m in candidates
    }
    return sorted(candidates, key=lambda m: combined[m.memory_id], reverse=True)
