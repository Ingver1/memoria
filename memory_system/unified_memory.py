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
from collections import deque
from collections.abc import MutableMapping, Sequence

# local
from dataclasses import dataclass
from typing import Any, Protocol


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
    ) -> Memory: ...

    async def list_recent(self, *, n: int = 20) -> Sequence[Memory]: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


ASYNC_TIMEOUT = 5  # seconds – safety net for accidental long‑running operations.

# Process-wide default store used when no explicit store is provided.
_DEFAULT_STORE: MemoryStoreProtocol | None = None

# ---------------------------------------------------------------------------
# Working memory helpers
# ---------------------------------------------------------------------------

WORKING_MEMORY_CAPACITY = 9
_working_memory: deque[Memory] = deque(maxlen=WORKING_MEMORY_CAPACITY)


def push_working_memory(memory: Memory) -> None:
    """Add *memory* to the working memory buffer."""

    _working_memory.append(memory)


def get_working_memory() -> Sequence[Memory]:
    """Return the current working memory contents in order of insertion."""

    return list(_working_memory)


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
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update text and/or metadata of an existing memory and return the new object.

    Args:
        memory_id (str): The memory identifier.
        text (str | None, optional): New text. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): New metadata. Defaults to None.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The updated memory object.
    """
    st = await _resolve_store(store)
    try:
        updated = await asyncio.wait_for(
            st.update_memory(memory_id, text=text, metadata=metadata), timeout=ASYNC_TIMEOUT
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
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Reinforce the importance of a memory by *amount* and return the updated object.

    Args:
        memory_id (str): The memory identifier.
        amount (float, optional): Amount to reinforce. Defaults to 0.1.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The reinforced memory object.
    """
    st = await _resolve_store(store)
    meta = {
        "last_accessed": _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat(),
    }
    try:
        updated = await asyncio.wait_for(
            st.update_memory(memory_id, importance_delta=amount, metadata=meta),
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
IMPORTANCE_WEIGHT = 1.0
EMOTIONAL_INTENSITY_WEIGHT = 1.0
VALENCE_POS_WEIGHT = 1.0
VALENCE_NEG_WEIGHT = 0.5


def _score_best(m: Memory) -> float:
    """Return the ranking score for a memory.

    ``valence`` contributes positively for pleasant memories and is
    penalised when negative using :data:`VALENCE_NEG_WEIGHT`.
    """

    valence_weight = VALENCE_POS_WEIGHT if m.valence >= 0 else VALENCE_NEG_WEIGHT
    return (
        IMPORTANCE_WEIGHT * m.importance
        + EMOTIONAL_INTENSITY_WEIGHT * m.emotional_intensity
        + valence_weight * m.valence
    )


async def list_best(
    n: int = 5,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return *n* most important memories ranked by score.

    Args:
        n (int, optional): Number of top memories. Defaults to 5.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Sequence[Memory]: List of best memories ordered by score where
        negative ``valence`` reduces the overall ranking.
    """
    st = await _resolve_store(store)
    try:
        candidates = await asyncio.wait_for(st.list_recent(n=max(n * 5, 20)), timeout=ASYNC_TIMEOUT)
        scored = sorted(candidates, key=_score_best, reverse=True)
    except Exception as e:
        logger.error("List best failed: %s", e)
        raise
    return scored[:n]


__all__ = [
    "Memory",
    "MemoryStoreProtocol",
    "add",
    "search",
    "delete",
    "update",
    "reinforce",
    "push_working_memory",
    "get_working_memory",
    "list_best",
    "list_recent",
    "set_default_store",
    "get_default_store",
]
