"""memory_system.core.store
=================================
Asynchronous SQLite-backed memory store with JSON1 metadata support and
connection pooling via **aiosqlite**. Designed to be injected through a
FastAPI lifespan context — no hidden singletons.
"""

from __future__ import annotations

# ────────────────────────── stdlib imports ──────────────────────────
import asyncio
import datetime as dt
import json
import logging
import uuid
from collections.abc import AsyncIterator

# ───────────────────────── local imports ───────────────────────────
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, cast

# ─────────────────────── third-party imports ───────────────────────
import aiosqlite

if TYPE_CHECKING:  # pragma: no cover - optional FastAPI import for type hints
    from fastapi import FastAPI, Request
    
logger = logging.getLogger(__name__)

###############################################################################
# Data model
###############################################################################

@dataclass(slots=True, frozen=True)
class Memory:
    """A single memory entry with optional emotional context."""

    id: str
    text: str
    created_at: dt.datetime = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc)
    )
    importance: float = 0.0  # 0..1
    valence: float = 0.0  # -1..1 emotional polarity
    emotional_intensity: float = 0.0  # 0..1 strength of emotion
    metadata: Dict[str, Any] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Memory):
            return NotImplemented
        return (
            self.id == other.id
            and self.text == other.text
            and self.importance == other.importance
            and self.valence == other.valence
            and self.emotional_intensity == other.emotional_intensity
            and self.metadata == other.metadata
        )
        
    @staticmethod
    def new(
        text: str,
        *,
        importance: float = 0.0,
        valence: float = 0.0,
        emotional_intensity: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Memory":
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        if not -1.0 <= valence <= 1.0:
            raise ValueError("valence must be between -1 and 1")
        if not 0.0 <= emotional_intensity <= 1.0:
            raise ValueError("emotional_intensity must be between 0 and 1")
            
        return Memory(
            id=str(uuid.uuid4()),
            text=text,
            created_at=dt.datetime.now(dt.timezone.utc),
            importance=importance,
            valence=valence,
            emotional_intensity=emotional_intensity,
            metadata=metadata or {},
        )

###############################################################################
# Store implementation
###############################################################################

class SQLiteMemoryStore:
    """Async store that leverages SQLite JSON1 for flexible metadata queries."""

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        text        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        importance  REAL DEFAULT 0,
        valence     REAL DEFAULT 0,
        emotional_intensity REAL DEFAULT 0,
        metadata    JSON
    );
    """

    def __init__(self, dsn: str | Path = "file:memories.db?mode=rwc", *, pool_size: int = 5) -> None:
        if isinstance(dsn, Path):
            self._path = dsn
            self._dsn = f"file:{dsn}?mode=rwc"
        else:
            if dsn.startswith("sqlite+sqlcipher:///"):
                path_part = dsn.split("sqlite+sqlcipher:///", 1)[1]
                self._dsn = f"file:{path_part}?mode=rwc"
                path_str = path_part
            elif dsn.startswith("sqlite:///"):
                path_part = dsn.split("sqlite:///", 1)[1]
                self._dsn = f"file:{path_part}?mode=rwc"
                path_str = path_part
            else:
                self._dsn = dsn
                if dsn.startswith("file:"):
                    path_str = dsn[5:].split("?", 1)[0]
                elif "://" in dsn:
                    path_str = dsn.split("://", 1)[1].split("?", 1)[0]
                else:
                    path_str = dsn
            self._path = Path(path_str)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._pool_size = pool_size
        self._pool: asyncio.LifoQueue[aiosqlite.Connection] = asyncio.LifoQueue(maxsize=pool_size)
        self._conn = object()  # placeholder for tests
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop
            self._loop = asyncio.new_event_loop()
        self._initialised: bool = False
        self._lock = asyncio.Lock()  # protects initialisation & pool resize
        self._created = 0  # number of currently open connections

    # ---------------------------------------------------------------------
    # Low‑level connection helpers
    # ---------------------------------------------------------------------
    async def _acquire(self) -> aiosqlite.Connection:
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            if self._created < self._pool_size:
                conn = await aiosqlite.connect(self._dsn, uri=True, timeout=30)
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                await conn.execute("PRAGMA synchronous=NORMAL")
                conn.row_factory = aiosqlite.Row
                self._created += 1
                return conn
            return await self._pool.get()

    async def _release(self, conn: aiosqlite.Connection) -> None:
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            self._created -= 1

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    async def initialise(self) -> None:
        """Create table / indices once per process."""
        if self._initialised:
            return
        async with self._lock:
            if self._initialised:
                return
            conn = await self._acquire()
            try:
                await conn.execute(self._CREATE_SQL)
                await conn.commit()
                self._initialised = True
            finally:
                await self._release(conn)
            logger.info("SQLiteMemoryStore initialised (dsn=%s)", self._dsn)

    async def aclose(self) -> None:
        """Close all pooled connections."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
            self._created = 0

    async def close(self) -> None:
        """Compatibility alias for ``aclose`` used in tests."""
        await self.aclose()

    # -------------------------------------
    async def add(self, mem: Memory) -> None:
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute(
                "INSERT INTO memories (id, text, created_at, importance, valence, emotional_intensity, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, json(?))",
                (
                    mem.id,
                    mem.text,
                    mem.created_at.isoformat(),
                    mem.importance,
                    mem.valence,
                    mem.emotional_intensity,
                    json.dumps(mem.metadata) if mem.metadata else "null",
                ),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            await conn.close()
            self._created -= 1
            raise
        else:
            await self._release(conn)

    async def get(self, memory_id: str) -> Optional[Memory]:
        await self.initialise()
        conn = await self._acquire()
        try:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            )
            row = await cursor.fetchone()
            return self._row_to_memory(row) if row else None
        finally:
            await self._release(conn)

    async def ping(self) -> None:
        """Simple connectivity check used by readiness probes."""
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("SELECT 1")
        finally:
            await self._release(conn)

    def _row_to_memory(self, row: aiosqlite.Row | Any) -> Memory:
        """Map a database row or a row-like object to a :class:`Memory`."""

        def _get(obj: Any, key: str) -> Any:
            try:
                return obj[key]
            except Exception:
                return getattr(obj, key)

        meta_raw = _get(row, "metadata")
        metadata = json.loads(meta_raw) if meta_raw not in (None, "null") else None
        
        return Memory(
            id=_get(row, "id"),
            text=_get(row, "text"),
            created_at=dt.datetime.fromisoformat(_get(row, "created_at")),
            importance=_get(row, "importance"),
            valence=_get(row, "valence"),
            emotional_intensity=_get(row, "emotional_intensity"),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    async def search(
        self,
        text_query: Optional[str] = None,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """Simple LIKE + JSON1 metadata search (no vectors here)."""
        await self.initialise()
        conn = await self._acquire()
        try:
            # build WHERE clause
            clauses: List[str] = []
            params: List[Any] = []
            if text_query:
                clauses.append("text LIKE ?")
                params.append(f"%{text_query}%")
            if metadata_filters:
                for key, val in metadata_filters.items():
                    clauses.append("json_extract(metadata, ?) = ?")
                    params.extend([f"$.{key}", val])

            # construct final SQL
            sql = (
                "SELECT id, text, created_at, importance, valence, emotional_intensity, metadata FROM memories"
            )
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            # execute and map results
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    async def list_recent(self, *, n: int = 20) -> List[Memory]:
        """Return the most recent *n* memories."""
        await self.initialise()
        conn = await self._acquire()
        try:
            cursor = await conn.execute(
                "SELECT id, text, created_at, importance, valence, emotional_intensity, metadata "
                "FROM memories ORDER BY created_at DESC LIMIT ?",
                (n,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
                await self._release(conn)

    async def add_memory(self, mem_obj: Any) -> None:
        """Add a memory object, accepting either :class:`Memory` or a similar object."""
        await self.initialise()
        if isinstance(mem_obj, Memory):
            mem_to_add = mem_obj
        else:
            mid = (
                getattr(mem_obj, "id", None)
                or getattr(mem_obj, "memory_id", None)
                or str(uuid.uuid4())
            )
            mtext = mem_obj.text
            mcreated = getattr(
                mem_obj,
                "created_at",
                dt.datetime.now(dt.timezone.utc),
            )
            mimportance = getattr(mem_obj, "importance", 0.0)
            mvalence = getattr(mem_obj, "valence", 0.0)
            mintensity = getattr(mem_obj, "emotional_intensity", 0.0)
            mmeta = getattr(mem_obj, "metadata", None) or {}
            mem_to_add = Memory(
                id=mid,
                text=mtext,
                created_at=mcreated,
                importance=mimportance,
                valence=mvalence,
                emotional_intensity=mintensity,
                metadata=mmeta,
            )
        await self.add(mem_to_add)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID."""
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await conn.commit()
        finally:
            await self._release(conn)

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Memory:
        """Update text and/or metadata of an existing memory and return the updated row."""
        await self.initialise()
        conn = await self._acquire()
        try:
            if text is not None:
                await conn.execute(
                    "UPDATE memories SET text = ? WHERE id = ?",
                    (text, memory_id),
                )
            if metadata is not None:
                await conn.execute(
                    "UPDATE memories SET metadata = json(?) WHERE id = ?",
                    (json.dumps(metadata), memory_id),
                )
            await conn.commit()
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = await cursor.fetchone()
            if not row:
                raise RuntimeError("Memory not found")
            return self._row_to_memory(row)
        finally:
            await self._release(conn)

    async def search_memory(
        self,
        query: str,
        k: int = 5,
        *,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Memory]:
        """Search memories by text and optional metadata filters (alias for :meth:`search`)."""
        return await self.search(text_query=query, metadata_filters=metadata_filter, limit=k)

###############################################################################
# FastAPI integration helpers (optional import‑time dep)
###############################################################################

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan_context(app: "FastAPI") -> AsyncIterator[None]:  # pragma: no cover
    """FastAPI lifespan function that attaches a SQLiteMemoryStore to ``app.state``."""

    store = SQLiteMemoryStore()
    await store.initialise()
    app.state.memory_store = store
    try:
        yield
    finally:
        await store.aclose()


def get_memory_store(request: "Request") -> SQLiteMemoryStore:  # pragma: no cover
    return cast(SQLiteMemoryStore, request.app.state.memory_store)
    
###############################################################################
# Singleton helper
###############################################################################

_STORE: SQLiteMemoryStore | None = None
_STORE_LOCK = asyncio.Lock()


async def get_store(path: str | Path | None = None) -> SQLiteMemoryStore:
    """Return process-wide :class:`SQLiteMemoryStore` singleton.

    The store is created on first use and cached for subsequent calls.  If
    *path* is provided on the first call, it is used as the SQLite file path.
    Later calls ignore the parameter and return the already-created instance.
    """

    global _STORE
    async with _STORE_LOCK:
        global _STORE
        if _STORE is not None and path is not None and _STORE._path != Path(path):
            await _STORE.aclose()
            _STORE = None
        if _STORE is None:
            dsn = f"file:{path}?mode=rwc" if path else "file:memories.db?mode=rwc"
            _STORE = SQLiteMemoryStore(dsn)
            await _STORE.initialise()
        assert _STORE is not None
        return _STORE

from memory_system.core.enhanced_store import (
    EnhancedMemoryStore,
    HealthComponent,
)  # Ensure EnhancedMemoryStore & HealthComponent are accessible via core.store

__all__ = [
    "Memory",
    "SQLiteMemoryStore",
    "get_store",
    "EnhancedMemoryStore",
    "HealthComponent",
]
