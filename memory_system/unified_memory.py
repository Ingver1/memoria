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
import logging
import uuid

# stdlib
from collections.abc import MutableMapping, Sequence

# local
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(slots=True)
class Memory:
    """Simple memory record used by helper functions."""

    memory_id: str
    text: str
    created_at: _dt.datetime
    valence: float = 0.0
    emotional_intensity: float = 0.0
    arousal: float = 0.0
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
    ) -> Memory: ...

    async def list_recent(self, *, n: int = 20) -> Sequence[Memory]: ...

    async def top_n_by_score(
        self, n: int, score_fn: Callable[[Memory], float]
    ) -> Sequence[Memory]: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


ASYNC_TIMEOUT = 5  # seconds – safety net for accidental long‑running operations.

# Process-wide default store used when no explicit store is provided.
_DEFAULT_STORE: MemoryStoreProtocol | None = None


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def add(
    text: str,
    *,
    valence: float = 0.0,
    emotional_intensity: float = 0.0,
    arousal: float = 0.0,
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
        arousal (float, optional): Arousal level. Defaults to 0.0.
        importance (float, optional): Importance score. Defaults to 0.0.
        episode_id (str | None, optional): Episode identifier. Defaults to None.
        modality (str, optional): Modality type. Defaults to "text".
        connections (MutableMapping[str, float] | None, optional): Connections. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): Arbitrary metadata. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The persisted memory object.
    """
    memory = Memory(
        memory_id=str(uuid.uuid4()),
        text=text,
        valence=valence,
        emotional_intensity=emotional_intensity,
        arousal=arousal,
        importance=importance,
        episode_id=episode_id,
        modality=modality,
        connections=dict(copy.deepcopy(connections)) if connections else None,
        metadata=dict(copy.deepcopy(metadata)) if metadata else {},
        created_at=_dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc),
    )

    st = await _resolve_store(store)
    try:
        await asyncio.wait_for(st.add_memory(memory), timeout=ASYNC_TIMEOUT)
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
    st = await _resolve_store(store)
    try:
        meta = dict(metadata_filter or {})
        meta.setdefault("modality", modality)
        results = await asyncio.wait_for(
            st.search_memory(query=query, k=k, metadata_filter=meta, level=level),
            timeout=ASYNC_TIMEOUT,
        )
        logger.debug("Search for '%s' returned %d result(s).", query, len(results))
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise
    return results


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
        logger.debug("Memory %s deleted.", memory_id)
    except Exception as e:
        logger.error("Delete failed: %s", e)
        raise


async def update(
    memory_id: str,
    *,
    text: str | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    valence_delta: float | None = None,
    emotional_intensity_delta: float | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update text and/or metadata of an existing memory and return the new object.

    Args:
        memory_id (str): The memory identifier.
        text (str | None, optional): New text. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): New metadata. Defaults to None.
        valence_delta (float | None, optional): Increment for emotional valence.
            Defaults to None.
        emotional_intensity_delta (float | None, optional): Increment for
            emotional intensity. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The updated memory object.
    """
    st = await _resolve_store(store)
    try:
        updated = await asyncio.wait_for(
            st.update_memory(
                memory_id,
                text=text,
                metadata=metadata,
                valence_delta=valence_delta,
                emotional_intensity_delta=emotional_intensity_delta,
            ),
            timeout=ASYNC_TIMEOUT,
        )
        logger.debug("Memory %s updated.", memory_id)
    except Exception as e:
        logger.error("Update failed: %s", e)
        raise
    return updated


async def reinforce(
    memory_id: str,
    amount: float = 0.1,
    *,
    valence_delta: float | None = None,
    intensity_delta: float | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Reinforce a memory's importance and optionally its emotional context.

    By default only the ``importance`` field is reinforced. Supplying
    ``valence_delta`` and/or ``intensity_delta`` applies the same deltas to
    the respective ``valence`` and ``emotional_intensity`` attributes.

    Args:
        memory_id (str): The memory identifier.
        amount (float, optional): Importance increment. Defaults to 0.1.
        valence_delta (float | None, optional): Change applied to ``valence``.
        intensity_delta (float | None, optional): Change applied to
            ``emotional_intensity``.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The reinforced memory object.
    """
    st = await _resolve_store(store)
    try:
        updated = await asyncio.wait_for(
            st.update_memory(
                memory_id,
                importance_delta=amount,
                valence_delta=valence_delta,
                emotional_intensity_delta=intensity_delta,
            ),
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
        arousal=getattr(m, "arousal", 0.0),
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
    weights: ListBestWeights | None = None,
    include_all: bool = False,
) -> Sequence[Memory]:
    """Return *n* most important memories ranked by score.

    Args:
        n (int, optional): Number of top memories. Defaults to 5.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.
        weights (ListBestWeights | None, optional): Weight configuration for
            ranking. Defaults to :class:`ListBestWeights`.
        include_all (bool, optional): When ``True`` the whole store is scanned
            using a priority queue instead of just the most recent entries.

    Returns:
        Sequence[Memory]: List of best memories ordered by score where
        negative ``valence`` reduces the overall ranking.
    """
    st = await _resolve_store(store)
    if weights is None:
        weights = ListBestWeights()
    try:
        if include_all:
            # Attempt to leverage store-level optimisation for full scans
            candidates = await asyncio.wait_for(
                st.top_n_by_score(n, lambda m: _score_best(m, weights)),
                timeout=ASYNC_TIMEOUT,
            )
            return [_ensure_memory(m) for m in candidates]

        candidates = await asyncio.wait_for(
            st.list_recent(n=max(n * 5, 20)),
            timeout=ASYNC_TIMEOUT,
        )
        scored = sorted(candidates, key=lambda m: _score_best(m, weights), reverse=True)
    except Exception as e:
        logger.error("List best failed: %s", e)
        raise
    return [_ensure_memory(m) for m in scored[:n]]


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
