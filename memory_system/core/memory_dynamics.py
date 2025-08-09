from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from memory_system.core.store import Memory
from memory_system.unified_memory import ListBestWeights


class SupportsDynamics(Protocol):
    async def update(self, memory_id: str, **kwargs: Any) -> Memory: ...
    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None: ...


def _default_weights() -> ListBestWeights:
    try:  # lazy import to avoid heavy deps at import time
        from memory_system.config.settings import get_settings

        cfg = get_settings()
        return ListBestWeights(**cfg.ranking.model_dump())
    except Exception:  # pragma: no cover - settings module optional
        return ListBestWeights()


def _decay_rate() -> float:
    """Return decay rate from settings or a default."""
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.config.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return max(dyn.decay_rate, 1e-9)
    except Exception:  # pragma: no cover - settings module optional
        return 30.0


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

        meta = {"last_accessed": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()}
        updated = await self.store.update(
            memory_id,
            importance_delta=amount,
            valence_delta=valence_delta,
            emotional_intensity_delta=intensity_delta,
            metadata=meta,
        )
        score = self.score(updated)
        await self.store.upsert_scores([(memory_id, score)])
        return updated

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score(self, memory: Memory, *, now: dt.datetime | None = None) -> float:
        """Return the time-decayed ranking score for *memory*."""
        valence_weight = self.weights.valence_pos if memory.valence >= 0 else self.weights.valence_neg
        imp = max(0.0, min(1.0, memory.importance))
        inten = max(0.0, min(1.0, memory.emotional_intensity))
        val = max(-1.0, min(1.0, memory.valence))
        base = self.weights.importance * imp + self.weights.emotional_intensity * inten + valence_weight * val
        last = self._last_accessed(memory)
        if now is None:
            now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        age_days = max(0.0, (now - last).total_seconds() / 86_400.0)
        return base * math.exp(-age_days / _decay_rate())

    @staticmethod
    def _last_accessed(memory: Memory) -> dt.datetime:
        if memory.metadata:
            ts = memory.metadata.get("last_accessed")
            if isinstance(ts, str):
                try:
                    return dt.datetime.fromisoformat(ts)
                except ValueError:
                    pass
        return memory.created_at

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
        base = 0.4 * importance + 0.3 * emotional_intensity + 0.3 * valence
        base = max(0.0, base)
        return base * math.exp(-age_days / _decay_rate())
