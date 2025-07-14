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

    async def add_memory(self, memory: Memory) -> None:
        ...

    async def search_memory(
        self, *, query: str, k: int = 5, metadata_filter: MutableMapping[str, Any] | None = None
    ) -> Sequence[Memory]:
        ...

    async def delete_memory(self, memory_id: str) -> None:
        ...

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: MutableMapping[str, Any] | None = None,
    ) -> Memory:
        ...

    async def list_recent(self, *, n: int = 20) -> Sequence[Memory]:
        ...

logger = logging.getLogger("memory_system.unified_memory")

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

    Parameters
        ----------
    text:
        Raw textual content of the memory.
    metadata:
        Arbitrary key/‑value mapping (JSON‑serialisable).  Reserved keys
        such as ``created_at`` or ``memory_id`` will be overwritten.
    store:
        Optional explicit store object implementing the protocol.  If
        *None* the process‑wide default is used.
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
        connections=dict(connections) if connections else None,
        metadata=dict(metadata) if metadata else {},
        created_at=_dt.datetime.now(_dt.timezone.utc),
    )

    st = await _resolve_store(store)
    await asyncio.wait_for(st.add_memory(memory), timeout=ASYNC_TIMEOUT)
    logger.debug("Memory %s added (%d chars).", memory.memory_id, len(text))
    return memory


async def search(
    query: str,
    k: int = 5,
    *,
    metadata_filter: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Semantic search across stored memories.

    Parameters
    ----------
    query:
        Search phrase.
    k:
        Maximum number of results.
    metadata_filter:
        Optional mapping – only memories whose metadata contains all
        specified keys/values will be considered.
    store:
        Explicit store object or *None* for the default.
    """
    st = await _resolve_store(store)
    results = await asyncio.wait_for(
        st.search_memory(query=query, k=k, metadata_filter=metadata_filter),
        timeout=ASYNC_TIMEOUT,
    )
    logger.debug("Search for '%s' returned %d result(s).", query, len(results))
    return results


async def delete(
    memory_id: str,
    *,
    store: MemoryStoreProtocol | None = None,
) -> None:
    """Delete a memory by ``memory_id`` if it exists."""
    st = await _resolve_store(store)
    await asyncio.wait_for(st.delete_memory(memory_id), timeout=ASYNC_TIMEOUT)
    logger.debug("Memory %s deleted.", memory_id)


async def update(
    memory_id: str,
    *,
    text: str | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Update text and/or metadata of an existing memory and return the new object."""
    st = await _resolve_store(store)
    updated = await asyncio.wait_for(
        st.update_memory(memory_id, text=text, metadata=metadata), timeout=ASYNC_TIMEOUT
    )
    logger.debug("Memory %s updated.", memory_id)
    return updated


async def reinforce(
    memory_id: str,
    amount: float = 0.1,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """Reinforce the importance of a memory by *amount* and return the updated object."""

    st = await _resolve_store(store)
    meta = {"importance_delta": amount, "last_accessed": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    updated = await asyncio.wait_for(
        st.update_memory(memory_id, metadata=meta), timeout=ASYNC_TIMEOUT
    )
    logger.debug("Memory %s reinforced by %.2f.", memory_id, amount)
    return updated


async def list_recent(
    n: int = 20,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return *n* most recently added memories in descending chronological order."""
    st = await _resolve_store(store)
    recent = await asyncio.wait_for(st.list_recent(n=n), timeout=ASYNC_TIMEOUT)
    logger.debug("Fetched %d recent memories.", len(recent))
    return recent


async def list_best(
    n: int = 5,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Sequence[Memory]:
    """Return *n* most important memories ranked by score."""
    st = await _resolve_store(store)
    candidates = await asyncio.wait_for(st.list_recent(n=max(n * 5, 20)), timeout=ASYNC_TIMEOUT)
    scored = sorted(
        candidates,
        key=lambda m: (m.importance + m.emotional_intensity + abs(m.valence)),
        reverse=True,
    )
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
