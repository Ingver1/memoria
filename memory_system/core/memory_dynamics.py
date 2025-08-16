from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from memory_system.core.store import Memory
from memory_system.unified_memory import ListBestWeights, last_accessed

log = logging.getLogger(__name__)

EPSILON = 1e-9


class SupportsDynamics(Protocol):
    async def update(self, memory_id: str, **kwargs: Any) -> Memory: ...
    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None: ...


def _default_weights() -> ListBestWeights:
    try:  # lazy import to avoid heavy deps at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        return ListBestWeights(
            **cfg.ranking.model_dump(exclude={"min_score", "use_cross_encoder", "adaptation"})
        )
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - settings module optional
        return ListBestWeights()


def _decay_rate() -> float:
    """Return decay rate from settings or a default."""
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return float(max(dyn.decay_rate, EPSILON))
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return 30.0


def _decay_law() -> str:
    """Return decay law from settings or default to exponential."""
    try:
        from memory_system.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return getattr(dyn, "decay_law", "exponential")
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return "exponential"


def _age_decay(age_days: float) -> float:
    """Return decay multiplier for an age in days."""
    rate = _decay_rate()
    law = _decay_law()
    if law == "logarithmic":
        return 1.0 / (1.0 + math.log1p(age_days) / rate)
    return math.exp(-age_days / rate)


def _decay_weights() -> tuple[float, float, float]:
    """Return decay weights from settings or defaults."""
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        weights = getattr(dyn, "decay_weights", None)
        if weights is None:
            raise AttributeError
        return (
            getattr(weights, "importance", 0.4),
            getattr(weights, "emotional_intensity", 0.3),
            getattr(weights, "valence", 0.3),
        )
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return 0.4, 0.3, 0.3


def _is_dev_mode() -> bool:
    """Return ``True`` when running in development profile."""
    try:  # Lazy import to avoid heavy deps at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        return getattr(cfg, "profile", "") == "development"
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return False


_RECENCY_TAU = 86_400.0


@dataclass
class MemoryDynamics:
    """Utilities for memory reinforcement, decay and scoring."""

    store: SupportsDynamics | None = None
    weights: ListBestWeights = field(default_factory=_default_weights)

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------

    async def reinforce(
        self,
        memory_id: str,
        amount: float = 0.1,
        *,
        valence_delta: float | None = None,
        intensity_delta: float | None = None,
    ) -> Memory:
        """Reinforce *memory_id* by ``amount`` and update its score."""
        if self.store is None:
            raise RuntimeError("store is required for reinforcement")

        now = dt.datetime.now(dt.UTC)
        meta = {"last_accessed": now.isoformat()}
        updated = await self.store.update(
            memory_id,
            importance_delta=amount,
            valence_delta=valence_delta,
            emotional_intensity_delta=intensity_delta,
            metadata=meta,
        )

        # Drop immediately if TTL expired
        ttl = getattr(updated, "ttl_seconds", None)
        if ttl is not None and (now - updated.created_at).total_seconds() > ttl:
            delete = getattr(self.store, "delete_memory", None)
            if callable(delete):
                await delete(memory_id)
            raise RuntimeError("memory expired")

        # Spaced repetition: update last_used and extend decay interval
        prev_decay = getattr(updated, "decay", 1.0) or 1.0
        new_decay = prev_decay * 1.5
        updated = await self.store.update(memory_id, last_used=now, decay=new_decay)

        # Recompute score for the updated memory and bump access counter
        score = cast("float", self.score(updated, return_parts=False))

        prev = 0
        try:
            prev = int((updated.metadata or {}).get("access_count", 0))
        except (ValueError, TypeError):
            prev = 0

        new_count = prev + 1
        updated = await self.store.update(memory_id, metadata={"access_count": new_count})
        await self.store.upsert_scores([(memory_id, score)])
        return updated

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score(
        self,
        memory: Memory,
        *,
        now: dt.datetime | None = None,
        return_parts: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        """
        Return the ranking score for *memory* with intensity decay.

        When ``return_parts`` is ``True`` the individual weighted components
        are returned alongside the final score as ``(score, parts)`` where
        ``parts`` is a mapping of component name to its contribution.
        """
        if now is None:
            now = dt.datetime.now(dt.UTC)

        ttl = getattr(memory, "ttl_seconds", None)
        if ttl is not None and (now - memory.created_at).total_seconds() > ttl:
            if return_parts:
                return 0.0, {"importance": 0.0, "emotional_intensity": 0.0, "valence": 0.0}
            return 0.0

        valence_weight = (
            self.weights.valence_pos if memory.valence >= 0 else self.weights.valence_neg
        )
        ema = 0.0
        if memory.metadata:
            try:
                ema = float(memory.metadata.get("ema_access", 0.0))
            except (ValueError, TypeError):
                ema = 0.0
        imp = max(0.0, min(1.0, memory.importance))
        inten = max(0.0, min(1.0, memory.emotional_intensity))
        val = max(-1.0, min(1.0, memory.valence))
        last = memory.last_used or last_accessed(memory)
        delta = max(0.0, (now - last).total_seconds())
        age_days = delta / 86_400.0
        if getattr(memory, "decay", None):
            age_days /= max(memory.decay or 1.0, EPSILON)
        decay = _age_decay(age_days)
        inten *= decay
        recency = math.exp(-delta / _RECENCY_TAU)

        imp_part = self.weights.importance * imp
        inten_part = self.weights.emotional_intensity * inten
        val_part = valence_weight * val
        # Apply recency to positive and negative parts asymmetrically so that
        # older memories never outrank newer ones regardless of sign.
        pos = imp_part + inten_part + max(0.0, val_part)
        neg = min(0.0, val_part)
        freq_part = self.weights.frequency * math.log1p(ema)
        score = pos * recency + (neg / max(recency, EPSILON)) + freq_part

        if _is_dev_mode():
            log.debug(
                "score parts for %s: importance=%.3f, intensity=%.3f, valence=%.3f -> %.3f",
                memory.id,
                imp_part,
                inten_part,
                val_part,
                score,
            )

        if return_parts:
            return (
                score,
                {
                    "importance": imp_part * recency,
                    "emotional_intensity": inten_part * recency,
                    "valence": (max(0.0, val_part) * recency) + (min(0.0, val_part) / max(recency, EPSILON)),
                    "frequency": freq_part,
                },
            )
        return score

    # ------------------------------------------------------------------
    # Decay scoring used for forgetting
    # ------------------------------------------------------------------

    @staticmethod
    def decay(
        *,
        importance: float,
        valence: float,
        emotional_intensity: float,
        age_days: float,
    ) -> float:
        """Return an age-aware retention score (higher → keep)."""
        importance = max(0.0, min(1.0, importance))
        valence = max(-1.0, min(1.0, valence))
        emotional_intensity = max(0.0, min(1.0, emotional_intensity))
        imp_w, inten_w, val_w = _decay_weights()
        base = imp_w * importance + inten_w * emotional_intensity + val_w * valence
        base = max(0.0, base)
        return base * _age_decay(age_days)
