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
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator

# ───────────────────────── local imports ───────────────────────────
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, MutableMapping, Optional, Sequence, cast

# ─────────────────────── third-party imports ───────────────────────
import aiosqlite
from memory_system.unified_memory import ListBestWeights

from .top_n_by_score_sql import build_top_n_by_score_sql

try:  # Optional libSQL support
    from libsql_client import create_client as _create_libsql_client
except Exception:  # pragma: no cover - libsql optional
    _create_libsql_client = None


class _LibSQLRow(dict):
    """Row object that supports both index and key access."""

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        super().__init__(zip(columns, values, strict=True))
        self._values = list(values)

    def __getitem__(self, item: Any) -> Any:  # type: ignore[override]
        if isinstance(item, int):
            return self._values[item]
        return super().__getitem__(item)


class _LibSQLCursor:
    def __init__(self, result: Any) -> None:
        cols = getattr(result, "columns", [])
        rows = getattr(result, "rows", [])
        self._rows = [_LibSQLRow(cols, row) for row in rows]
        self._iter = iter(self._rows)

    async def fetchone(self) -> _LibSQLRow | None:
        try:
            return next(self._iter)
        except StopIteration:
            return None

    async def fetchall(self) -> list[_LibSQLRow]:
        return list(self._rows)


class _LibSQLConnection:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> _LibSQLCursor:
        res = await self._client.execute(sql, params or [])
        return _LibSQLCursor(res)

    async def executescript(self, script: str) -> None:
        for stmt in filter(None, (s.strip() for s in script.split(";"))):
            await self._client.execute(stmt)

    async def commit(self) -> None:  # pragma: no cover - libsql auto commits
        return None

    async def rollback(self) -> None:  # pragma: no cover - libsql auto commits
        """Provide rollback API for parity with :mod:`aiosqlite`.

        libSQL performs implicit commits, so a rollback operation is a
        no-op.  The method exists to mirror the interface of
        :class:`aiosqlite.Connection` which the rest of the store expects.
        """
        return None

    async def close(self) -> None:
        await self._client.close()


if TYPE_CHECKING:  # pragma: no cover - optional FastAPI import for type hints
    from fastapi import FastAPI, Request

    from memory_system.core.index import FaissHNSWIndex

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 10_000


def _safe_json(data: Dict[str, Any] | None) -> str:
    if not data:
        return "null"
    try:
        dumped = json.dumps(data)
        json.loads(dumped)
    except (TypeError, ValueError) as exc:
        raise TypeError("metadata must be JSON serializable") from exc
    return dumped


###############################################################################
# Data model
###############################################################################


@dataclass(slots=True, frozen=True)
class Memory:
    """A single memory entry with optional emotional context."""

    id: str
    text: str
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    importance: float = 0.0  # 0..1
    valence: float = 0.0  # -1..1 emotional polarity
    emotional_intensity: float = 0.0  # 0..1 strength of emotion
    metadata: Dict[str, Any] | None = None
    level: int = 0
    episode_id: str | None = None
    modality: str = "text"
    connections: Dict[str, float] | None = None

    def __eq__(self, other: object) -> bool:
        """Compare two memories by all fields."""
        if not isinstance(other, Memory):
            return NotImplemented
        return (
            self.id == other.id
            and self.text == other.text
            and self.importance == other.importance
            and self.valence == other.valence
            and self.emotional_intensity == other.emotional_intensity
            and self.metadata == other.metadata
            and self.level == other.level
            and self.episode_id == other.episode_id
            and self.modality == other.modality
            and self.connections == other.connections
        )

    @staticmethod
    def new(
        text: str,
        *,
        importance: float = 0.0,
        valence: float = 0.0,
        emotional_intensity: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        level: int = 0,
        episode_id: str | None = None,
        modality: str = "text",
        connections: Optional[Dict[str, float]] = None,
    ) -> "Memory":
        """Return a new :class:`Memory` with a generated UUID."""
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
            level=level,
            episode_id=episode_id,
            modality=modality,
            connections=connections,
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
        level       INTEGER DEFAULT 0,
        episode_id  TEXT,
        modality    TEXT DEFAULT 'text',
        connections JSON,
        metadata    JSON
    );
    """

    def __init__(
        self,
        dsn: str | Path = "file:memories.db?mode=rwc",
        *,
        pool_size: int = 5,
        wal: bool = True,
        synchronous: str = "NORMAL",
        page_size: int | None = None,
        cache_size: int | None = None,
    ) -> None:
        """Initialise the store with a SQLite or libSQL DSN and pool size."""
        self._use_libsql = False
        if isinstance(dsn, Path):
            self._path = dsn
            self._dsn = f"file:{dsn}?mode=rwc"
        else:
            if dsn.startswith("libsql://"):
                self._dsn = dsn
                self._use_libsql = True
                self._path = Path(".")
            elif dsn.startswith("sqlite+sqlcipher:///"):
                path_part = dsn.split("sqlite+sqlcipher:///", 1)[1]
                self._dsn = f"file:{path_part}?mode=rwc"
                path_str = path_part
                self._path = Path(path_str)
            elif dsn.startswith("sqlite:///"):
                path_part = dsn.split("sqlite:///", 1)[1]
                self._dsn = f"file:{path_part}?mode=rwc"
                path_str = path_part
                self._path = Path(path_str)
            else:
                self._dsn = dsn
                if dsn.startswith("file:"):
                    path_str = dsn[5:].split("?", 1)[0]
                elif "://" in dsn:
                    path_str = dsn.split("://", 1)[1].split("?", 1)[0]
                else:
                    path_str = dsn
                self._path = Path(path_str)
        if not self._use_libsql:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._pool_size = pool_size
        self._wal = wal
        self._synchronous = synchronous.upper()
        self._page_size = page_size
        self._cache_size = cache_size
        self._pool: asyncio.LifoQueue[Any] = asyncio.LifoQueue(maxsize=pool_size)
        self._conn = object()  # placeholder for tests
        self._acquired: set[Any] = set()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:  # no running loop
            self._loop = asyncio.new_event_loop()
        self._initialised: bool = False
        self._lock = asyncio.Lock()  # protects initialisation & pool resize
        self._created = 0  # number of currently open connections
        # Hooks executed after a successful commit
        self._commit_hooks: list[Callable[[], Any]] = []

    # ---------------------------------------------------------------------
    # Low‑level connection helpers
    # ---------------------------------------------------------------------
    async def _acquire(self) -> Any:
        """Obtain a connection from the pool, creating one if necessary."""
        if self._use_libsql:
            try:
                conn = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                if _create_libsql_client is None:  # pragma: no cover - optional dep
                    raise RuntimeError("libsql-client is not installed") from None
                raw = await _create_libsql_client(self._dsn)
                conn = _LibSQLConnection(raw)
                self._created += 1
            self._acquired.add(conn)
            return conn
        try:
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            if self._created < self._pool_size:
                conn = await aiosqlite.connect(self._dsn, uri=True, timeout=30, check_same_thread=False)
                if self._page_size is not None:
                    await conn.execute(f"PRAGMA page_size={self._page_size}")
                if self._cache_size is not None:
                    await conn.execute(f"PRAGMA cache_size={self._cache_size}")
                if self._wal:
                    await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                await conn.execute(f"PRAGMA synchronous={self._synchronous}")
                conn.row_factory = aiosqlite.Row
                self._created += 1
            else:
                conn = await self._pool.get()
        self._acquired.add(conn)
        return conn

    async def _release(self, conn: Any) -> None:
        """Return ``conn`` to the pool or close it if the pool is full."""
        self._acquired.discard(conn)
        try:
            self._pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            self._created -= 1

    # ------------------------------------------------------------------
    # Commit hooks
    # ------------------------------------------------------------------
    def add_commit_hook(self, hook: Callable[[], Any]) -> None:
        """Register a hook executed after each successful commit."""
        self._commit_hooks.append(hook)

    def remove_commit_hook(self, hook: Callable[[], Any]) -> None:
        """Remove a previously registered commit hook."""
        try:
            self._commit_hooks.remove(hook)
        except ValueError:
            pass

    async def _run_commit_hooks(self) -> None:
        for hook in list(self._commit_hooks):
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("commit hook failed")

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
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_scores (
                        memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                        score     REAL NOT NULL
                    )
                    """
                )
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_scores_score ON memory_scores(score)")
                # Ensure FTS virtual table and triggers exist
                await conn.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text);
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid = old.rowid;
                    END;
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid = old.rowid;
                        INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
                    END;
                    """
                )
                # Migration: backfill FTS table if empty or out of sync
                cur = await conn.execute("SELECT count(*) FROM memories")
                mem_count = (await cur.fetchone())[0]
                cur = await conn.execute("SELECT count(*) FROM memories_fts")
                fts_count = (await cur.fetchone())[0]
                if mem_count != fts_count:
                    await conn.execute("DELETE FROM memories_fts")
                    await conn.execute("INSERT INTO memories_fts(rowid, text) SELECT rowid, text FROM memories")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level)")
                await conn.commit()
                self._initialised = True
            finally:
                await self._release(conn)
            logger.info("SQLiteMemoryStore initialised (dsn=%s)", self._dsn)

    async def aclose(self) -> None:
        """Close all pooled connections and any acquired ones."""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        for conn in list(self._acquired):
            await conn.close()
        self._acquired.clear()
        self._created = 0
        self._initialised = False

    async def close(self) -> None:
        """Compatibility alias for ``aclose`` used in tests."""
        await self.aclose()

    # -------------------------------------
    async def add(self, mem: Memory) -> None:
        """Persist a new :class:`Memory` to the database."""
        await self.initialise()
        if len(mem.text) > MAX_TEXT_LENGTH:
            raise ValueError("text exceeds maximum length")
        conn = await self._acquire()
        try:
            await conn.execute(
                "INSERT INTO memories (id, text, created_at, importance, valence, emotional_intensity, level, episode_id, modality, connections, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?))",
                (
                    mem.id,
                    mem.text,
                    mem.created_at.isoformat(),
                    mem.importance,
                    mem.valence,
                    mem.emotional_intensity,
                    mem.level,
                    mem.episode_id,
                    mem.modality,
                    _safe_json(mem.connections),
                    _safe_json(mem.metadata),
                ),
            )
            await conn.commit()
            await self._run_commit_hooks()
        except Exception:
            await conn.rollback()
            await conn.close()
            self._created -= 1
            raise
        else:
            await self._release(conn)

    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None:
        """Insert multiple memories using a single transaction.

        Parameters
        ----------
        memories:
            Sequence of :class:`Memory` objects to insert.
        batch_size:
            Number of records to send per ``executemany`` call.
        """

        if not memories:
            return

        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("BEGIN")
            sql = (
                "INSERT INTO memories (id, text, created_at, importance, valence, emotional_intensity, level, episode_id, modality, connections, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?))"
            )
            batch: list[tuple[Any, ...]] = []
            for mem in memories:
                if len(mem.text) > MAX_TEXT_LENGTH:
                    raise ValueError("text exceeds maximum length")
                batch.append(
                    (
                        mem.id,
                        mem.text,
                        mem.created_at.isoformat(),
                        mem.importance,
                        mem.valence,
                        mem.emotional_intensity,
                        mem.level,
                        mem.episode_id,
                        mem.modality,
                        _safe_json(mem.connections),
                        _safe_json(mem.metadata),
                    )
                )
                if len(batch) >= batch_size:
                    await conn.executemany(sql, batch)
                    batch.clear()
            if batch:
                await conn.executemany(sql, batch)
            await conn.commit()
            await self._run_commit_hooks()
        except Exception:
            await conn.rollback()
            await conn.close()
            self._created -= 1
            raise
        else:
            await self._release(conn)

    async def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch a memory by its ID."""
        await self.initialise()
        conn = await self._acquire()
        try:
            cursor = await conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
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
        conn_raw = _get(row, "connections")
        connections = json.loads(conn_raw) if conn_raw not in (None, "null") else None

        return Memory(
            id=_get(row, "id"),
            text=_get(row, "text"),
            created_at=dt.datetime.fromisoformat(_get(row, "created_at")),
            importance=_get(row, "importance"),
            valence=_get(row, "valence"),
            emotional_intensity=_get(row, "emotional_intensity"),
            metadata=metadata,
            level=_get(row, "level"),
            episode_id=_get(row, "episode_id"),
            modality=_get(row, "modality"),
            connections=connections,
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
        level: int | None = None,
        offset: int = 0,
    ) -> List[Memory]:
        """Full-text + JSON1 metadata search (no vectors here)."""
        await self.initialise()
        conn = await self._acquire()
        try:
            params: List[Any] = []
            if text_query:
                sql = (
                    "SELECT m.id, m.text, m.created_at, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ?"
                )
                params.append(text_query)
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality"}:
                            sql += f" AND m.{key} = ?"
                            params.append(val)
                        else:
                            sql += " AND json_extract(m.metadata, ?) = ?"
                            params.extend([f"$.{key}", val])
                if level is not None:
                    sql += " AND m.level = ?"
                    params.append(level)
                sql += " ORDER BY bm25(memories_fts) LIMIT ? OFFSET ?"
            else:
                clauses: List[str] = []
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality"}:
                            clauses.append(f"{key} = ?")
                            params.append(val)
                        else:
                            clauses.append("json_extract(metadata, ?) = ?")
                            params.extend([f"$.{key}", val])
                if level is not None:
                    clauses.append("level = ?")
                    params.append(level)
                sql = "SELECT id, text, created_at, importance, valence, emotional_intensity, level, episode_id, modality, connections, metadata FROM memories"
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cursor = await conn.execute(sql, params)
            results: list[Memory] = []
            if hasattr(cursor, "_cursor") and hasattr(cursor._cursor, "fetchmany"):
                batch_size = min(limit, 1000)
                while True:
                    batch = await asyncio.to_thread(cursor._cursor.fetchmany, batch_size)
                    if not batch:
                        break
                    results.extend(self._row_to_memory(r) for r in batch)
            else:
                rows = await cursor.fetchall()
                results = [self._row_to_memory(r) for r in rows]
            return results
        finally:
            await self._release(conn)

    async def search_iter(
        self,
        text_query: Optional[str] = None,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int | None = None,
        level: int | None = None,
        min_level: int | None = None,
        final: bool | None = None,
        chunk_size: int = 1000,
    ) -> AsyncIterator[list[Memory]]:
        """Yield search results in chunks to keep memory usage bounded.

        Results are streamed using a SQLite cursor and ``fetchmany`` to avoid
        loading the entire result set into memory. Additional filters allow
        callers to restrict the range of levels via ``min_level`` and to
        include only memories marked as final or not via ``final``.
        """

        await self.initialise()
        conn = await self._acquire()
        try:
            params: list[Any] = []
            if text_query:
                sql = (
                    "SELECT m.id, m.text, m.created_at, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ?"
                )
                params.append(text_query)
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality"}:
                            sql += f" AND m.{key} = ?"
                            params.append(val)
                        else:
                            sql += " AND json_extract(m.metadata, ?) = ?"
                            params.extend([f"$.{key}", val])
                if level is not None:
                    sql += " AND m.level = ?"
                    params.append(level)
                if min_level is not None:
                    sql += " AND m.level >= ?"
                    params.append(min_level)
                if final is not None:
                    if final:
                        sql += " AND json_extract(m.metadata, '$.final') = 1"
                    else:
                        sql += (
                            " AND (json_extract(m.metadata, '$.final') IS NULL "
                            "OR json_extract(m.metadata, '$.final') = 0)"
                        )
                sql += " ORDER BY bm25(memories_fts)"
            else:
                clauses: list[str] = []
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality"}:
                            clauses.append(f"{key} = ?")
                            params.append(val)
                        else:
                            clauses.append("json_extract(metadata, ?) = ?")
                            params.extend([f"$.{key}", val])
                if level is not None:
                    clauses.append("level = ?")
                    params.append(level)
                if min_level is not None:
                    clauses.append("level >= ?")
                    params.append(min_level)
                if final is not None:
                    if final:
                        clauses.append("json_extract(metadata, '$.final') = 1")
                    else:
                        clauses.append(
                            "(json_extract(metadata, '$.final') IS NULL OR json_extract(metadata, '$.final') = 0)"
                        )
                sql = (
                    "SELECT id, text, created_at, importance, valence, emotional_intensity, level, "
                    "episode_id, modality, connections, metadata FROM memories"
                )
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                sql += " ORDER BY created_at DESC"

            cursor = await conn.execute(sql, params)
            fetched = 0
            try:
                while True:
                    batch_size = chunk_size
                    if limit is not None:
                        remaining = limit - fetched
                        if remaining <= 0:
                            break
                        batch_size = min(batch_size, remaining)
                    rows = await asyncio.to_thread(cursor._cursor.fetchmany, batch_size)
                    if not rows:
                        break
                    yield [self._row_to_memory(r) for r in rows]
                    fetched += len(rows)
            finally:
                await asyncio.to_thread(cursor._cursor.close)
        finally:
            await self._release(conn)

    async def list_recent(self, *, n: int = 20, level: int | None = None) -> List[Memory]:
        """Return the most recent *n* memories, optionally filtered by level."""
        await self.initialise()
        conn = await self._acquire()
        try:
            params: tuple[Any, ...]
            sql = (
                "SELECT id, text, created_at, importance, valence, emotional_intensity, level, "
                "episode_id, modality, connections, metadata FROM memories"
            )
            if level is not None:
                sql += " WHERE level = ? ORDER BY created_at DESC LIMIT ?"
                params = (level, n)
            else:
                sql += " ORDER BY created_at DESC LIMIT ?"
                params = (n,)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None:
        """Insert or update precomputed ranking *scores*.

        Parameters
        ----------
        scores:
            Sequence of ``(memory_id, score)`` pairs.
        """
        if not scores:
            return
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("BEGIN")
            await conn.executemany(
                "INSERT INTO memory_scores(memory_id, score) VALUES (?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET score = excluded.score",
                scores,
            )
            await conn.commit()
            await self._run_commit_hooks()
        except Exception:
            await conn.rollback()
            await conn.close()
            self._created -= 1
            raise
        else:
            await self._release(conn)

    async def top_n_by_score(
        self,
        n: int,
        *,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
        weights: ListBestWeights | None = None,
        ids: Sequence[str] | None = None,
    ) -> List[Memory]:
        """Return ``n`` memories ordered by ranking score.

        Parameters
        ----------
        n:
            Number of memories to return.
        level:
            Optional exact level filter.
        metadata_filter:
            Mapping of metadata key/value pairs that must all match.
        weights:
            When provided, ranking is computed on the fly using these
            weighting coefficients.  When omitted, precomputed scores from
            the ``memory_scores`` table are used.
        ids:
            Optional iterable of memory IDs to constrain the result to.
        """

        await self.initialise()
        conn = await self._acquire()
        try:
            if weights is None:
                clauses: list[str] = []
                params: list[Any] = []
                if level is not None:
                    clauses.append("m.level = ?")
                    params.append(level)
                if metadata_filter:
                    for key, val in metadata_filter.items():
                        if key in {"episode_id", "modality"}:
                            clauses.append(f"m.{key} = ?")
                            params.append(val)
                        else:
                            clauses.append("json_extract(m.metadata, ?) = ?")
                            params.extend([f"$.{key}", val])
                if ids:
                    placeholders = ", ".join(["?"] * len(ids))
                    clauses.append(f"m.id IN ({placeholders})")
                    params.extend(ids)
                sql = (
                    "SELECT m.id, m.text, m.created_at, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata "
                    "FROM memory_scores s JOIN memories m ON m.id = s.memory_id"
                )
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                sql += " ORDER BY s.score DESC LIMIT ?"
                params.append(n)
            else:
                sql, params = build_top_n_by_score_sql(
                    n,
                    weights,
                    level=level,
                    metadata_filter=metadata_filter,
                    ids=ids,
                )
            cursor = await conn.execute(sql, params)
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
            mid = getattr(mem_obj, "id", None) or getattr(mem_obj, "memory_id", None) or str(uuid.uuid4())
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
            mlevel = getattr(mem_obj, "level", 0)
            mepisode = getattr(mem_obj, "episode_id", None)
            mmodality = getattr(mem_obj, "modality", "text")
            mconnections = getattr(mem_obj, "connections", None)
            mem_to_add = Memory(
                id=mid,
                text=mtext,
                created_at=mcreated,
                importance=mimportance,
                valence=mvalence,
                emotional_intensity=mintensity,
                metadata=mmeta,
                level=mlevel,
                episode_id=mepisode,
                modality=mmodality,
                connections=mconnections,
            )
        await self.add(mem_to_add)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID."""
        await self.initialise()
        conn = await self._acquire()
        try:
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await conn.commit()
            await self._run_commit_hooks()
        finally:
            await self._release(conn)

    async def delete(self, memory_id: str) -> None:
        """Alias for :meth:`delete_memory` to satisfy :class:`MetaStore` interface."""
        await self.delete_memory(memory_id)

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: Dict[str, Any] | None = None,
        importance: float | None = None,
        importance_delta: float | None = None,
        valence: float | None = None,
        valence_delta: float | None = None,
        emotional_intensity: float | None = None,
        emotional_intensity_delta: float | None = None,
    ) -> Memory:
        """
        Update text, scores and metadata.

        ``importance``, ``valence`` and ``emotional_intensity`` set new values
        for the respective fields (after clamping to their ranges).  The
        ``*_delta`` counterparts add the provided amount to the current value,
        also clamped.  Metadata is shallow‑merged as JSON.  Ranges:
        importance ∈ [0,1], emotional_intensity ∈ [0,1], valence ∈ [-1,1].
        """
        await self.initialise()
        conn = await self._acquire()
        try:
            if text is not None:
                if len(text) > MAX_TEXT_LENGTH:
                    raise ValueError("text exceeds maximum length")
                await conn.execute(
                    "UPDATE memories SET text = ? WHERE id = ?",
                    (text, memory_id),
                )

            if importance is not None:
                await conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    (importance, memory_id),
                )
            elif importance_delta is not None:
                await conn.execute(
                    "UPDATE memories SET importance = MAX(0.0, MIN(1.0, importance + ?)) WHERE id = ?",
                    (importance_delta, memory_id),
                )

            if valence is not None:
                await conn.execute(
                    "UPDATE memories SET valence = ? WHERE id = ?",
                    (valence, memory_id),
                )
            elif valence_delta is not None:
                await conn.execute(
                    "UPDATE memories SET valence = MAX(-1.0, MIN(1.0, valence + ?)) WHERE id = ?",
                    (valence_delta, memory_id),
                )

            if emotional_intensity is not None:
                await conn.execute(
                    "UPDATE memories SET emotional_intensity = ? WHERE id = ?",
                    (emotional_intensity, memory_id),
                )
            elif emotional_intensity_delta is not None:
                await conn.execute(
                    "UPDATE memories "
                    "SET emotional_intensity = MAX(0.0, MIN(1.0, emotional_intensity + ?)) "
                    "WHERE id = ?",
                    (emotional_intensity_delta, memory_id),
                )

            if metadata is not None and metadata:
                cursor = await conn.execute(
                    "SELECT metadata FROM memories WHERE id = ?",
                    (memory_id,),
                )
                row = await cursor.fetchone()
                if not row:
                    raise RuntimeError("Memory not found")
                existing = json.loads(row[0] or "{}")
                if existing is None:
                    existing = {}
                existing.update(metadata)
                await conn.execute(
                    "UPDATE memories SET metadata = json(?) WHERE id = ?",
                    (_safe_json(existing), memory_id),
                )

            await conn.commit()
            await self._run_commit_hooks()
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

    async def update(self, memory_id: str, **kwargs: Any) -> Memory:
        """Alias for :meth:`update_memory` used by :class:`MetaStore`."""
        return await self.update_memory(memory_id, **kwargs)

    async def search_memory(
        self,
        query: str,
        k: int = 5,
        *,
        metadata_filter: Optional[Dict[str, Any]] = None,
        level: int | None = None,
    ) -> List[Memory]:
        """Search memories by text and optional metadata filters (alias for :meth:`search`)."""
        return await self.search(text_query=query, metadata_filters=metadata_filter, limit=k, level=level)


###############################################################################
# FastAPI integration helpers (optional import‑time dep)
###############################################################################

from contextlib import asynccontextmanager


@asynccontextmanager
async def persist_index_on_commit(
    store: "SQLiteMemoryStore",
    index: "FaissHNSWIndex",
    path: str,
) -> AsyncIterator[None]:
    """Ensure FAISS index is saved whenever the store commits."""

    async def _save() -> None:
        await asyncio.to_thread(index.save, path)

    store.add_commit_hook(_save)
    try:
        yield
    finally:
        store.remove_commit_hook(_save)


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
    """Return the SQLiteMemoryStore attached to the FastAPI request."""
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
        # Providing a new *path* closes the existing store and creates a
        # fresh singleton instance pointing at the new location.
        if _STORE is not None and path is not None and _STORE._path != Path(path):
            await _STORE.aclose()
            _STORE = None
        if _STORE is None:
            dsn = f"file:{path}?mode=rwc" if path else "file:memories.db?mode=rwc"
            _STORE = SQLiteMemoryStore(dsn)
            await _STORE.initialise()
        assert _STORE is not None
        return _STORE


try:  # pragma: no cover - optional dependency
    from memory_system.core.enhanced_store import (
        EnhancedMemoryStore,
        HealthComponent,
    )  # Ensure EnhancedMemoryStore & HealthComponent are accessible via core.store
except Exception:  # pragma: no cover - optional dependency missing
    EnhancedMemoryStore = None  # type: ignore[assignment]
    HealthComponent = None  # type: ignore[assignment]

__all__ = [
    "Memory",
    "SQLiteMemoryStore",
    "persist_index_on_commit",
    "get_store",
]
if EnhancedMemoryStore is not None and HealthComponent is not None:
    __all__.extend(["EnhancedMemoryStore", "HealthComponent"])
