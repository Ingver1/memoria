"""memory_system.unified_memory
================================
High‑level, **framework‑agnostic** helper functions that wrap the lower
level storage / vector components.  Nothing in here depends on FastAPI
or any other web framework – the goal is to let notebooks, background
jobs, or other services reuse the same persistence logic without pulling
in heavy HTTP deps.

Notes
-----
* All functions are **async**.  They accept an optional ``store`` kwarg –
  any object that implements ``add_memory``, ``search_memory``,
  ``delete_memory`` and ``update_metadata``.  If not supplied the helper
  tries to obtain the application‑scoped store via
  :pyfunc:`memory_system.core.store.get_memory_store`.
* Docstrings follow **PEP 257** and type hints are 100 % complete so that
  MyPy / Ruff‑strict pass cleanly.
"""

from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import functools
import logging
import math
import uuid

# stdlib
from collections.abc import MutableMapping, Sequence
from types import SimpleNamespace

# local
from dataclasses import dataclass
from typing import Any, Protocol

from memory_system.utils.cache import SmartCache


@dataclass(slots=True)
class Memory:
    """Simple memory record used by helper functions."""

    memory_id: str
    text: str
    created_at: _dt.datetime
    valence: float = 0.0
    emotional_intensity: float = 0.0
    importance: float = 0.0
    episode_id: str | None = None
    modality: str = "text"
    connections: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


class MemoryStoreProtocol(Protocol):
    """Minimal protocol expected from a memory store."""

    async def add_memory(self, memory: Memory) -> None: ...

    async def search_memory(
        self, *, query: str, k: int = 5, metadata_filter: MutableMapping[str, Any] | None = None
    ) -> Sequence[Memory]: ...

    async def delete_memory(self, memory_id: str) -> None: ...

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: MutableMapping[str, Any] | None = None,
        importance: float | None = None,
        importance_delta: float | None = None,
        valence: float | None = None,
        valence_delta: float | None = None,
        emotional_intensity: float | None = None,
        emotional_intensity_delta: float | None = None,
    ) -> Memory:
        """Update fields or scores of a memory.

        Absolute parameters overwrite stored values while ``*_delta`` fields
        apply relative adjustments.
        """

        ...

    async def list_recent(self, *, n: int = 20) -> Sequence[Memory]: ...

    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None: ...

    async def top_n_by_score(
        self,
        n: int,
        *,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
        weights: ListBestWeights | None = None,
        ids: Sequence[str] | None = None,
    ) -> Sequence[Memory]: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


ASYNC_TIMEOUT = 5  # seconds – safety net for accidental long‑running operations.

# Process-wide default store used when no explicit store is provided.
_DEFAULT_STORE: MemoryStoreProtocol | None = None

_CACHE: SmartCache | None = None


def _get_cache() -> SmartCache:
    """Return module-wide cache instance configured from settings."""

    global _CACHE
    if _CACHE is None:
        try:  # local import to avoid heavy config dependency at import time
            from memory_system.config.settings import get_settings

            cfg = get_settings()
            _CACHE = SmartCache(max_size=cfg.cache.size, ttl=cfg.cache.ttl_seconds)
        except Exception:  # pragma: no cover - settings module optional
            _CACHE = SmartCache()
    return _CACHE


def set_default_store(store: MemoryStoreProtocol) -> None:
    """Register *store* as the fallback for all helper functions."""
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


def get_default_store() -> MemoryStoreProtocol | None:
    """Return the currently registered default store, if any."""
    return _DEFAULT_STORE


async def _resolve_store(
    store: MemoryStoreProtocol | None = None,
) -> MemoryStoreProtocol:
    """Return a concrete store instance.

    The rules are:
    1. If *store* is given → use it.
    2. Else return the process‑wide default registered via :func:`set_default_store`.
    """

    if store is not None:
        return store

    resolved = get_default_store()
    if resolved is None:
        raise RuntimeError("Memory store has not been initialised.")
    return resolved


def _get_dynamics() -> SimpleNamespace:
    """Return dynamics configuration or sane defaults."""
    try:  # lazy import to avoid heavy deps at import time
        from memory_system.config.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return dyn
    except Exception:  # pragma: no cover - settings module optional
        return SimpleNamespace(initial_intensity=0.0, reinforce_delta=0.1, decay_rate=30.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def add(
    text: str,
    *,
    valence: float = 0.0,
    emotional_intensity: float | None = None,
    importance: float = 0.0,
    episode_id: str | None = None,
    modality: str = "text",
    connections: MutableMapping[str, float] | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Persist a *text* record with optional *metadata* and return a **Memory**.

    Args:
        text (str): Raw textual content of the memory.
        valence (float, optional): Emotional valence. Defaults to 0.0.
        emotional_intensity (float, optional): Intensity of emotion. Defaults to 0.0.
        importance (float, optional): Importance score. Defaults to 0.0.
        episode_id (str | None, optional): Episode identifier. Defaults to None.
        modality (str, optional): Modality type. Defaults to "text".
        connections (MutableMapping[str, float] | None, optional): Connections. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): Arbitrary metadata. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The persisted memory object.
    """
    now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    meta: MutableMapping[str, Any] = dict(copy.deepcopy(metadata) if metadata else {})
    meta.setdefault("last_accessed", now.isoformat())
    dyn = _get_dynamics()
    if emotional_intensity is None:
        emotional_intensity = dyn.initial_intensity
    valence = max(-1.0, min(1.0, valence))
    emotional_intensity = max(0.0, min(1.0, emotional_intensity))
    importance = max(0.0, min(1.0, importance))
    memory = Memory(
        memory_id=str(uuid.uuid4()),
        text=text,
        valence=valence,
        emotional_intensity=emotional_intensity,
        importance=importance,
        episode_id=episode_id,
        modality=modality,
        connections=dict(copy.deepcopy(connections)) if connections else None,
        metadata=meta,
        created_at=now,
    )

    st = await _resolve_store(store)
    try:
        await asyncio.wait_for(st.add_memory(memory), timeout=ASYNC_TIMEOUT)
        weights = _get_ranking_weights()
        score = _score_decay(memory, weights)
        await asyncio.wait_for(
            st.upsert_scores([(memory.memory_id, score)]),
            timeout=ASYNC_TIMEOUT,
        )
        logger.debug("Memory %s added (%d chars).", memory.memory_id, len(text))
    except Exception as e:
        logger.error("Failed to add memory: %s", e)
        raise
    return memory


async def search(
    query: str,
    k: int = 5,
    *,
    metadata_filter: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
    modality: str = "text",
    level: int | None = None,
) -> Sequence[Memory]:
    """Semantic search across stored memories.

    Args:
        query (str): Search phrase.
        k (int, optional): Maximum number of results. Defaults to 5.
        metadata_filter (MutableMapping[str, Any] | None, optional): Metadata filter. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Sequence[Memory]: List of matching memories.
    """
    cache = _get_cache()
    meta = dict(metadata_filter or {})
    meta.setdefault("modality", modality)
    key = repr((query, k, tuple(sorted(meta.items())), level))
    cached = cache.get(key)
    if cached is not None:
        return cached

    st = await _resolve_store(store)
    try:
        results = await asyncio.wait_for(
            st.search_memory(query=query, k=k, metadata_filter=meta, level=level),
            timeout=ASYNC_TIMEOUT,
        )
        results_list = list(results)
        cache.put(key, results_list)
        logger.debug("Search for '%s' returned %d result(s).", query, len(results_list))
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise
    return results_list


async def delete(
    memory_id: str,
    *,
    store: MemoryStoreProtocol | None = None,
) -> None:
    """Delete a memory by ``memory_id`` if it exists.

    Args:
        memory_id (str): The memory identifier.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.
    """
    st = await _resolve_store(store)
    try:
        await asyncio.wait_for(st.delete_memory(memory_id), timeout=ASYNC_TIMEOUT)
        _get_cache().clear()
        logger.debug("Memory %s deleted.", memory_id)
    except Exception as e:
        logger.error("Delete failed: %s", e)
        raise


async def update(
    memory_id: str,
    *,
    text: str | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    importance: float | None = None,
    importance_delta: float | None = None,
    valence: float | None = None,
    valence_delta: float | None = None,
    emotional_intensity: float | None = None,
    emotional_intensity_delta: float | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update text, metadata or scoring fields of an existing memory.

    This helper allows changing the ``importance`` score directly without
    calling :func:`reinforce` by supplying either ``importance`` or
    ``importance_delta``.

    ``last_accessed`` in the memory's metadata is automatically set to the
    current UTC timestamp.  Absolute ``valence`` and ``emotional_intensity``
    values may be supplied to overwrite existing ones while the ``*_delta``
    counterparts adjust the stored values.

    Args:
        memory_id (str): The memory identifier.
        text (str | None, optional): New text. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): New metadata. Defaults to None.
        importance (float | None, optional): New importance value. Defaults to None.
        importance_delta (float | None, optional): Increment for importance. Defaults to None.
        valence (float | None, optional): New emotional valence. Defaults to None.
        valence_delta (float | None, optional): Increment for emotional valence.
            Defaults to None.
        emotional_intensity (float | None, optional): New emotional intensity.
            Defaults to None.
        emotional_intensity_delta (float | None, optional): Increment for
            emotional intensity. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The updated memory object.
    """
    st = await _resolve_store(store)
    meta: MutableMapping[str, Any] = dict(copy.deepcopy(metadata) if metadata else {})
    meta["last_accessed"] = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()
    try:
        updated = await asyncio.wait_for(
            st.update_memory(
                memory_id,
                text=text,
                metadata=meta,
                importance=importance,
                importance_delta=importance_delta,
                valence=valence,
                valence_delta=valence_delta,
                emotional_intensity=emotional_intensity,
                emotional_intensity_delta=emotional_intensity_delta,
            ),
            timeout=ASYNC_TIMEOUT,
        )
        weights = _get_ranking_weights()
        score = _score_decay(updated, weights)
        await asyncio.wait_for(
            st.upsert_scores([(memory_id, score)]),
            timeout=ASYNC_TIMEOUT,
        )
        _get_cache().clear()
        logger.debug("Memory %s updated.", memory_id)
    except Exception as e:
        logger.error("Update failed: %s", e)
        raise
    return updated


async def reinforce(
    memory_id: str,
    amount: float | None = None,
    *,
    valence_delta: float | None = None,
    intensity_delta: float | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Reinforce a memory's importance and optionally its emotional context.

    The memory's metadata receives an updated ``last_accessed`` timestamp.

    By default only the ``importance`` field is reinforced. Supplying
    ``valence_delta`` and/or ``intensity_delta`` applies the same deltas to
    the respective ``valence`` and ``emotional_intensity`` attributes.

    Args:
        memory_id (str): The memory identifier.
        amount (float | None, optional): Importance increment. Defaults to configured value.
        valence_delta (float | None, optional): Change applied to ``valence``.
        intensity_delta (float | None, optional): Change applied to
            ``emotional_intensity``.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The reinforced memory object.
    """
    st = await _resolve_store(store)
    meta = {"last_accessed": _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()}
    dyn = _get_dynamics()
    if amount is None:
        amount = dyn.reinforce_delta
    amount = max(-1.0, min(1.0, amount))
    if valence_delta is not None:
        valence_delta = max(-1.0, min(1.0, valence_delta))
    if intensity_delta is not None:
        intensity_delta = max(-1.0, min(1.0, intensity_delta))
    try:
        updated = await asyncio.wait_for(
            st.update_memory(
                memory_id,
                importance_delta=amount,
                valence_delta=valence_delta,
                emotional_intensity_delta=intensity_delta,
                metadata=meta,
            ),
            timeout=ASYNC_TIMEOUT,
        )
        weights = _get_ranking_weights()
        score = _score_decay(updated, weights)
        await asyncio.wait_for(
            st.upsert_scores([(memory_id, score)]),
            timeout=ASYNC_TIMEOUT,
        )
        logger.debug("Memory %s reinforced by %.2f.", memory_id, amount)
    except Exception as e:
        logger.error("Reinforce failed: %s", e)
        raise
    return updated


async def list_recent(
    n: int = 20,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return *n* most recently added memories in descending chronological order.

    Args:
        n (int, optional): Number of memories. Defaults to 20.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Sequence[Memory]: List of recent memories.
    """
    st = await _resolve_store(store)
    try:
        recent = await asyncio.wait_for(st.list_recent(n=n), timeout=ASYNC_TIMEOUT)
        logger.debug("Fetched %d recent memories.", len(recent))
    except Exception as e:
        logger.error("List recent failed: %s", e)
        raise
    return recent


# Weights used when ranking memories via :func:`list_best`.
# Positive valence is treated as a benefit while negative valence is
# penalised with a reduced weight so that strongly negative memories
# need additional importance or intensity to surface.


@dataclass(slots=True)
class ListBestWeights:
    """Configuration for weighting memory attributes when ranking."""

    importance: float = 1.0
    emotional_intensity: float = 1.0
    valence_pos: float = 1.0
    valence_neg: float = 0.5


def _get_ranking_weights() -> ListBestWeights:
    """Load ranking weights from configuration if available."""
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.config.settings import get_settings

        cfg = get_settings()
        return ListBestWeights(**cfg.ranking.model_dump())
    except Exception:  # pragma: no cover - settings module optional
        return ListBestWeights()


def _score_best(m: Memory, weights: ListBestWeights) -> float:
    """Return the ranking score for a memory.

    ``valence`` contributes positively for pleasant memories and is
    penalised when negative via ``weights.valence_neg``.
    """

    valence_weight = weights.valence_pos if m.valence >= 0 else weights.valence_neg
    return (
        weights.importance * m.importance
        + weights.emotional_intensity * m.emotional_intensity
        + valence_weight * m.valence
    )


def _last_accessed(m: Memory) -> _dt.datetime:
    """Return the last accessed timestamp for *m*.

    Falls back to ``m.created_at`` when the ``metadata`` lacks the
    ``last_accessed`` key or contains an invalid timestamp.
    """

    if m.metadata:
        ts = m.metadata.get("last_accessed")
        if isinstance(ts, str):
            try:
                return _dt.datetime.fromisoformat(ts)
            except ValueError:
                pass
    return m.created_at


def _score_decay(m: Memory, weights: ListBestWeights) -> float:
    """Return a time-decayed ranking score for *m*.

    The base score is derived from :func:`_score_best` and is exponentially
    decayed over roughly one month based on the elapsed time since
    ``last_accessed``.
    """

    base = _score_best(m, weights)
    last = _last_accessed(m)
    age_days = max(
        0.0,
        (_dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc) - last).total_seconds() / 86_400.0,
    )
    rate = max(_get_dynamics().decay_rate, 1e-9)
    return base * math.exp(-age_days / rate)


def _ensure_memory(m: Any) -> Memory:
    """Coerce *m* into a :class:`Memory` instance if needed."""
    if isinstance(m, Memory):
        return m
    return Memory(
        memory_id=getattr(m, "memory_id", m.id),
        text=m.text,
        created_at=m.created_at,
        valence=getattr(m, "valence", 0.0),
        emotional_intensity=getattr(m, "emotional_intensity", 0.0),
        importance=getattr(m, "importance", 0.0),
        episode_id=getattr(m, "episode_id", None),
        modality=getattr(m, "modality", "text"),
        connections=getattr(m, "connections", None),
        metadata=getattr(m, "metadata", None),
    )


async def list_best(
    n: int = 5,
    *,
    store: MemoryStoreProtocol | None = None,
    level: int | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    weights: ListBestWeights | None = None,
) -> Sequence[Memory]:
    """
    Return the `n` memories ranked by score.

    Parameters
    ----------
    n:
        Number of memories to return.
    store:
        Optional explicit memory store.
    level:
        Optional level filter applied before ranking.
    metadata_filter:
        Optional metadata constraints (exact match) applied before ranking.
    weights:
        If ``None`` (default) the store returns memories ordered by its
        precomputed ``memory_scores`` table.  When provided, the store
        computes the ranking on the fly using these weighting coefficients.
    """
    st = await _resolve_store(store)
    try:
        candidates = await asyncio.wait_for(
            st.top_n_by_score(
                n,
                level=level,
                metadata_filter=metadata_filter,
                weights=weights,
            ),
            timeout=ASYNC_TIMEOUT,
        )
    except Exception as e:
        logger.error("list_best failed: %s", e)
        raise

    return [_ensure_memory(m) for m in candidates]


__all__ = [
    "Memory",
    "MemoryStoreProtocol",
    "ListBestWeights",
    "add",
    "search",
    "delete",
    "update",
    "reinforce",
    "list_best",
    "list_recent",
    "set_default_store",
    "get_default_store",
]
