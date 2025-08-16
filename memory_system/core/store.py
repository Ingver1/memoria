"""
memory_system.core.store
=================================
Asynchronous SQLite-backed memory store with JSON1 metadata support and
connection pooling via **aiosqlite**. Designed to be injected through a
FastAPI lifespan context — no hidden singletons.
"""

from __future__ import annotations

# ────────────────────────── stdlib imports ──────────────────────────
import asyncio
import contextlib
import datetime as dt
import inspect
import json
import logging
import math
import os
import re
import unicodedata
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, MutableMapping, Sequence
from contextlib import asynccontextmanager

# ───────────────────────── local imports ───────────────────────────
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast


# In Python 3.12 ``Path.readlink`` returns a :class:`Path` object.  Some tests
# expect the older string behaviour, so normalise here for compatibility.
def _readlink(self: Path) -> str:  # pragma: no cover - simple shim
    return os.readlink(self)


_PathAny = cast(Any, Path)  # help mypy accept monkeypatching class attribute
_PathAny.readlink = cast("Callable[[Path], str]", _readlink)
from urllib.parse import parse_qs

# ─────────────────────── third-party imports ───────────────────────
from memory_system._vendor import aiosqlite
from memory_system.unified_memory import (
    ListBestWeights,
    MemoryStoreProtocol,
    SearchResults,
)
from memory_system.utils.blake import blake3_hex
from memory_system.utils.cache import SmartCache
from memory_system.utils.loop import get_loop
from memory_system.utils.security import EncryptionManager, SecretStr

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from cryptography.fernet import Fernet as _Fernet

Fernet: type[_Fernet] | None
try:
    from cryptography.fernet import Fernet as _FernetCls
except ImportError:  # pragma: no cover - cryptography optional
    Fernet = None
else:
    Fernet = _FernetCls

from .event_log import EventLog
from .keyword_attention import STOP_WORDS, TOKEN_RE
from .top_n_by_score_sql import build_top_n_by_score_sql, validate_metadata_key

try:  # Optional libSQL support
    from libsql_client import create_client as _create_libsql_client
except ImportError:  # pragma: no cover - libsql optional
    _create_libsql_client = None


class _LibSQLRow(dict[str, Any]):
    """Row object that supports both index and key access."""

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        super().__init__(zip(columns, values, strict=True))
        self._values = list(values)

    def __getitem__(self, item: str | int) -> Any:
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
        """
        Provide rollback API for parity with :mod:`aiosqlite`.

        libSQL performs implicit commits, so a rollback operation is a
        no-op.  The method exists to mirror the interface of
        :class:`aiosqlite.Connection` which the rest of the store expects.
        """
        return

    async def close(self) -> None:
        await self._client.close()


if TYPE_CHECKING:  # pragma: no cover - optional FastAPI import for type hints
    from fastapi import FastAPI, Request

    from memory_system.core.index import FaissHNSWIndex

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("memory_system.audit")

MAX_TEXT_LENGTH = 10_000


def _safe_json(data: dict[str, Any] | None) -> str:
    if not data:
        return "null"
    try:
        dumped = json.dumps(data)
        json.loads(dumped)
    except (TypeError, ValueError) as exc:
        raise TypeError("metadata must be JSON serializable") from exc
    return dumped


def _normalize_for_hash(text: str) -> str:
    """Normalize text for hashing: NFKC, lower, strip, collapse spaces."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


###############################################################################
# Data model
###############################################################################


@dataclass(slots=True, frozen=True)
class _MemoryImpl:
    """A single memory entry with optional emotional context."""

    id: str
    text: str
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    valid_from: dt.datetime | None = None
    valid_to: dt.datetime | None = None
    tx_from: dt.datetime | None = None
    tx_to: dt.datetime | None = None
    importance: float = 0.0  # 0..1
    valence: float = 0.0  # -1..1 emotional polarity
    emotional_intensity: float = 0.0  # 0..1 strength of emotion
    metadata: dict[str, Any] | None = None
    level: int = 0
    episode_id: str | None = None
    modality: str = "text"
    connections: dict[str, float] | None = None
    memory_type: Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] = (
        "episodic"
    )
    pinned: bool = False
    ttl_seconds: int | None = None
    last_used: dt.datetime | None = None
    success_score: float | None = None
    decay: float | None = None

    def __post_init__(self) -> None:
        """Ensure mandatory metadata fields are present."""
        meta = dict(self.metadata or {})
        meta.setdefault("trust_score", 1.0)
        meta.setdefault("error_flag", False)
        object.__setattr__(self, "metadata", meta)
        vf = self.valid_from or self.created_at
        vt = self.valid_to or dt.datetime.max.replace(tzinfo=dt.UTC)
        tf = self.tx_from or self.created_at
        tt = self.tx_to or dt.datetime.max.replace(tzinfo=dt.UTC)
        object.__setattr__(self, "valid_from", vf)
        object.__setattr__(self, "valid_to", vt)
        object.__setattr__(self, "tx_from", tf)
        object.__setattr__(self, "tx_to", tt)

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
            and self.memory_type == other.memory_type
            and self.pinned == other.pinned
            and self.ttl_seconds == other.ttl_seconds
            and self.last_used == other.last_used
            and self.success_score == other.success_score
            and self.decay == other.decay
        )

    # Compatibility attribute -------------------------------------------------
    #
    # Some historical parts of the test-suite referenced the ``memory_id``
    # attribute instead of ``id``.  The primary API now exposes ``id`` but we
    # keep ``memory_id`` as a read-only alias for backwards compatibility.

    @property
    def memory_id(self) -> str:  # pragma: no cover - simple delegation
        return self.id

    @staticmethod
    def new(
        text: str,
        *,
        importance: float = 0.0,
        valence: float = 0.0,
        emotional_intensity: float = 0.0,
        metadata: dict[str, Any] | None = None,
        level: int = 0,
        episode_id: str | None = None,
        modality: str = "text",
        connections: dict[str, float] | None = None,
        memory_type: Literal[
            "sensory", "working", "episodic", "semantic", "skill", "lesson"
        ] = "episodic",
        pinned: bool = False,
        ttl_seconds: int | None = None,
        last_used: dt.datetime | None = None,
        success_score: float | None = None,
        decay: float | None = None,
    ) -> Memory:
        """Return a new :class:`Memory` with a generated UUID."""
        importance = max(0.0, min(1.0, importance))
        valence = max(-1.0, min(1.0, valence))
        emotional_intensity = max(0.0, min(1.0, emotional_intensity))

        return _MemoryImpl(
            id=str(uuid.uuid4()),
            text=text,
            created_at=dt.datetime.now(dt.UTC),
            importance=importance,
            valence=valence,
            emotional_intensity=emotional_intensity,
            metadata=metadata or {},
            level=level,
            episode_id=episode_id,
            modality=modality,
            connections=connections,
            memory_type=memory_type,
            pinned=pinned,
            ttl_seconds=ttl_seconds,
            last_used=last_used,
            success_score=0.0 if success_score is None else success_score,
            decay=0.0 if decay is None else decay,
        )


@dataclass(slots=True, frozen=True)
class Case:
    """Reusable problem-solving experience."""

    id: str
    problem: str
    plan: str
    outcome: str | None = None
    evaluation: float | None = None
    embedding: list[float] | None = None

    @staticmethod
    def new(
        problem: str,
        plan: str,
        *,
        outcome: str | None = None,
        evaluation: float | None = None,
        embedding: list[float] | None = None,
    ) -> Case:
        """Return a new :class:`Case` with generated UUID."""
        return Case(
            id=str(uuid.uuid4()),
            problem=problem,
            plan=plan,
            outcome=outcome,
            evaluation=evaluation,
            embedding=embedding,
        )


###############################################################################
# Store implementation
###############################################################################


ENVELOPE_MANAGER = EncryptionManager()

# For type checking we alias the public ``Memory`` name to the unified
# representation so methods implementing ``MemoryStoreProtocol`` match its
# signatures.  At runtime we expose the local implementation defined above.
# Expose the local implementation under the public name
Memory = _MemoryImpl  # type: ignore[assignment]


class SQLiteMemoryStore(MemoryStoreProtocol):
    """Async store that leverages SQLite JSON1 for flexible metadata queries."""

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        text        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        valid_from  TEXT NOT NULL,
        valid_to    TEXT NOT NULL,
        tx_from     TEXT NOT NULL,
        tx_to       TEXT NOT NULL,
        importance  REAL DEFAULT 0,
        valence     REAL DEFAULT 0,
        emotional_intensity REAL DEFAULT 0,
        level       INTEGER DEFAULT 0,
        episode_id  TEXT,
        modality    TEXT DEFAULT 'text',
        connections JSON,
        metadata    JSON,
        memory_type TEXT DEFAULT 'episodic',
        lang        TEXT DEFAULT '',
        source      TEXT DEFAULT '',
        pinned      INTEGER DEFAULT 0,
        ttl_seconds INTEGER,
        last_used   TEXT DEFAULT CURRENT_TIMESTAMP,
        success_score REAL DEFAULT 0,
        decay       REAL DEFAULT 0,
        content_hash TEXT NOT NULL,
        access_count INTEGER DEFAULT 1,
        last_access_ts REAL,
        ciphertext  BLOB,
        nonce       BLOB,
        cek_wrapped BLOB,
        kek_id      TEXT
    );
    """

    _CREATE_CASES_SQL = """
    CREATE TABLE IF NOT EXISTS cases (
        id TEXT PRIMARY KEY,
        problem TEXT NOT NULL,
        plan TEXT NOT NULL,
        outcome TEXT,
        evaluation REAL,
        embedding TEXT
    );
    """

    def __init__(
        self,
        dsn: str | Path = "file:memories.db?mode=rwc",
        *,
        # A single writer connection keeps the simplified aiosqlite stub and
        # SQLite's coarse locking semantics from fighting under load.
        # Additional read-only connections may be pooled via ``read_pool_size``.
        pool_size: int = 1,
        read_pool_size: int | None = None,
        wal: bool = True,
        synchronous: str = "FULL",
        page_size: int | None = None,
        cache_size: int | None = None,
        mmap_size: int | None = None,
        busy_timeout: int = 5_000,
        wal_interval: float = 60.0,
        wal_checkpoint_writes: int = 1_000,
        cipher_secret: SecretStr | None = None,
    ) -> None:
        """Initialise the store with a SQLite or libSQL DSN and pool sizes."""
        self._use_libsql = False
        self._cipher_secret: SecretStr | None = None
        self._using_sqlcipher = False
        self._write_enabled = True
        self._db_ok = True
        self._event_log: EventLog | None = None
        if isinstance(dsn, Path):
            self._path = dsn
            self._dsn = f"file:{dsn}?mode=rwc"
        elif dsn.startswith("libsql://"):
            self._dsn = dsn
            self._use_libsql = True
            self._path = Path()
            wal = False  # remote libSQL handles WAL/checkpointing
        elif dsn.startswith("sqlite+sqlcipher:///"):
            path_part = dsn.split("sqlite+sqlcipher:///", 1)[1]
            if "?" in path_part:
                path_str, query = path_part.split("?", 1)
                params = parse_qs(query)
                secret = params.get("cipher_secret", [None])[0]
                if secret:
                    cipher_secret = SecretStr(secret)
                if "mode=" not in query:
                    query = f"{query}&mode=rwc"
                self._dsn = f"file:{path_str}?{query}"
            else:
                path_str = path_part
                self._dsn = f"file:{path_str}?mode=rwc"
            self._path = Path(path_str)
            self._using_sqlcipher = True
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
        if self._use_libsql:
            self._read_dsn = self._dsn
        else:
            if "mode=" in self._dsn:
                self._read_dsn = re.sub(r"mode=[^&]+", "mode=ro", self._dsn)
            else:
                sep = "&" if "?" in self._dsn else "?"
                self._read_dsn = f"{self._dsn}{sep}mode=ro"
            self._path.parent.mkdir(parents=True, exist_ok=True)
        # ``pool_size`` is retained for backward compatibility; when provided
        # it specifies the size of the read-only pool.  ``read_pool_size`` takes
        # precedence when explicitly passed.
        self._read_pool_size = read_pool_size if read_pool_size is not None else pool_size
        self._wal = wal
        self._synchronous = synchronous.upper()
        self._page_size = page_size
        self._cache_size = cache_size
        self._mmap_size = mmap_size
        self._busy_timeout = busy_timeout
        self._wal_interval = wal_interval
        self._wal_checkpoint_writes = wal_checkpoint_writes
        self._cipher_secret = cipher_secret
        self._writes_since_checkpoint = 0
        self._wal_event = asyncio.Event()
        self._read_pool: asyncio.LifoQueue[Any] = asyncio.LifoQueue(maxsize=self._read_pool_size)
        self._pool = self._read_pool
        self._writer_conn: Any | None = None
        self._writer_lock = asyncio.Lock()
        self._acquired: set[Any] = set()
        # ``SQLiteMemoryStore`` may be instantiated outside of a running event
        # loop (e.g. during test setup).  The previous implementation created a
        # brand new loop in that case and stored it on ``self._loop``.  Any
        # background tasks scheduled on this private loop would later be awaited
        # from the test's loop during teardown, resulting in "Task attached to a
        # different loop" errors.  Instead we defer grabbing the loop until we
        # actually need to schedule a task; when no loop is running yet we simply
        # record ``None`` and fetch the current loop later.
        try:
            self._loop: asyncio.AbstractEventLoop | None = get_loop()
        except RuntimeError:  # no running loop
            self._loop = None
        self._initialised: bool = False
        self._initialising: asyncio.Task[Any] | None = None
        self._lock = asyncio.Lock()  # protects initialisation & pool resize
        self._read_created = 0  # number of currently open read connections
        # Hooks executed after a successful commit
        self._commit_hooks: list[Callable[[], Any]] = []
        # Token DF/IDF cache
        self._df_cache: SmartCache | None = None
        # Idempotency cache for API requests
        try:
            from memory_system.settings import get_settings

            cfg = get_settings()
            self._idempotency_cache = SmartCache(max_size=cfg.cache.size, ttl=cfg.cache.ttl_seconds)
        except Exception:
            self._idempotency_cache = SmartCache()
        self._doc_count = 0
        self._wal_checkpoint_task: asyncio.Task[None] | None = None
        self._tasks: list[asyncio.Task[Any]] = []
        self._stopping = asyncio.Event()

    def _event_logger(self) -> EventLog:
        if self._event_log is None:
            self._event_log = EventLog(self._dsn)
        return self._event_log

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------

    def get_idempotent(self, key: str) -> Any | None:
        """Return cached result for *key* if present."""
        return self._idempotency_cache.get(key)

    def set_idempotent(self, key: str, value: Any) -> None:
        """Store *value* under *key* in the idempotency cache."""
        self._idempotency_cache.put(key, value)

    # ---------------------------------------------------------------------
    # Low‑level connection helpers
    # ---------------------------------------------------------------------
    async def _get_connection(self, *, read_only: bool = False) -> Any:
        """Create a new SQLite connection with tuned PRAGMAs."""
        dsn = self._read_dsn if read_only else self._dsn
        conn = await aiosqlite.connect(dsn, uri=True, timeout=5, check_same_thread=False)
        try:
            if self._cipher_secret and self._cipher_secret.get_secret_value():
                await conn.execute("PRAGMA key = ?", (self._cipher_secret.get_secret_value(),))
            if not read_only and self._wal:
                # Enable WAL first and then set the durability level.
                await conn.execute("PRAGMA journal_mode=WAL")

            if self._wal:
                # ``NORMAL`` avoids the fsync on each transaction while still
                # providing crash consistency when a checkpoint runs.
                await conn.execute("PRAGMA synchronous=NORMAL")
            else:
                await conn.execute(f"PRAGMA synchronous={self._synchronous}")

            if self._page_size is not None:
                # Has effect only before tables are created or after VACUUM.
                await conn.execute(f"PRAGMA page_size={int(self._page_size)}")
            if self._cache_size is not None:
                await conn.execute(f"PRAGMA cache_size={int(self._cache_size)}")
            await conn.execute("PRAGMA temp_store=MEMORY")
            if self._mmap_size is not None and not self._using_sqlcipher:
                try:
                    await conn.execute(f"PRAGMA mmap_size={int(self._mmap_size)}")
                except Exception:  # pragma: no cover - platform specific
                    logger.warning("failed to set mmap_size", exc_info=True)
            await conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout)}")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("SELECT 1")
            conn.row_factory = aiosqlite.Row
        except Exception:
            with contextlib.suppress(Exception):
                await conn.close()
            raise
        return conn

    async def _acquire(self, *, write: bool = False) -> Any:
        """Obtain a connection for reading or writing."""
        if not self._initialised and asyncio.current_task() is not self._initialising:
            await self.initialise()
        if write:
            await self._writer_lock.acquire()
            if self._writer_conn is None:
                try:
                    if self._use_libsql:
                        if _create_libsql_client is None:  # pragma: no cover - optional dep
                            raise RuntimeError("libsql-client is not installed") from None
                        raw = await _create_libsql_client(self._dsn)
                        self._writer_conn = _LibSQLConnection(raw)
                    else:
                        self._writer_conn = await self._get_connection()
                        self._schedule_wal_checkpoint()
                except Exception:
                    self._writer_lock.release()
                    raise
            self._acquired.add(self._writer_conn)
            return self._writer_conn
        if self._use_libsql:
            try:
                conn = self._read_pool.get_nowait()
            except asyncio.QueueEmpty:
                if _create_libsql_client is None:  # pragma: no cover - optional dep
                    raise RuntimeError("libsql-client is not installed") from None
                raw = await _create_libsql_client(self._dsn)
                conn = _LibSQLConnection(raw)
                self._read_created += 1
            self._acquired.add(conn)
            return conn
        try:
            conn = self._read_pool.get_nowait()
        except asyncio.QueueEmpty:
            if self._read_created < self._read_pool_size:
                try:
                    conn = await self._get_connection(read_only=True)
                except Exception:
                    self._db_ok = False
                    self._write_enabled = False
                    with contextlib.suppress(Exception):
                        await conn.close()
                    raise
                self._read_created += 1
            else:
                conn = await self._read_pool.get()
        self._acquired.add(conn)
        return conn

    async def _run_wal_checkpoint(self) -> None:
        try:
            conn = await aiosqlite.connect(self._dsn, uri=True, timeout=5, check_same_thread=False)
            try:
                if self._cipher_secret and self._cipher_secret.get_secret_value():
                    await conn.execute("PRAGMA key = ?", (self._cipher_secret.get_secret_value(),))
                await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._writes_since_checkpoint = 0
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()
        except asyncio.CancelledError:
            # Task cancelled during shutdown; ignore to allow clean exit
            pass
        except Exception:
            logger.exception("wal checkpoint failed")

    async def _wal_checkpoint_loop(self) -> None:
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wal_event.wait(), timeout=self._wal_interval)
            self._wal_event.clear()
            await self._run_wal_checkpoint()

    def _schedule_wal_checkpoint(self) -> None:
        """Start a background task to periodically checkpoint the WAL file."""
        if not self._wal:
            return
        if self._wal_checkpoint_task is None:
            if self._loop is None:
                self._loop = get_loop()
            self._wal_checkpoint_task = self._loop.create_task(self._wal_checkpoint_loop())
            self._tasks.append(self._wal_checkpoint_task)
        if self._writes_since_checkpoint >= self._wal_checkpoint_writes:
            self._wal_event.set()

    async def _release(self, conn: Any) -> None:
        """Return ``conn`` to the appropriate pool."""
        self._acquired.discard(conn)
        if conn is self._writer_conn:
            self._writer_lock.release()
            return
        try:
            self._read_pool.put_nowait(conn)
        except asyncio.QueueFull:
            await conn.close()
            self._read_created -= 1

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Any]:
        """Acquire a connection and handle commit/rollback automatically."""
        conn = await self._acquire(write=True)
        try:
            yield conn
            await conn.commit()
        except Exception:
            if hasattr(conn, "rollback"):
                with contextlib.suppress(Exception):
                    await conn.rollback()
            await conn.close()
            if conn is self._writer_conn:
                self._writer_conn = None
            else:
                self._read_created -= 1
            raise
        else:
            await self._release(conn)

    def _ensure_writable(self) -> None:
        if not self._write_enabled:
            raise RuntimeError("writes disabled")

    # ------------------------------------------------------------------
    # Commit hooks
    # ------------------------------------------------------------------
    def add_commit_hook(self, hook: Callable[[], Any]) -> None:
        """Register a hook executed after each successful commit."""
        self._commit_hooks.append(hook)

    def remove_commit_hook(self, hook: Callable[[], Any]) -> None:
        """Remove a previously registered commit hook."""
        with contextlib.suppress(ValueError):
            self._commit_hooks.remove(hook)

    async def _run_commit_hooks(self) -> None:
        self._writes_since_checkpoint += 1
        if (
            self._wal_checkpoint_writes
            and self._writes_since_checkpoint >= self._wal_checkpoint_writes
        ):
            await self._run_wal_checkpoint()
        for hook in list(self._commit_hooks):
            result = hook()
            if inspect.isawaitable(result):
                await result

    async def rotate_key(self, new_key: str | SecretStr) -> None:
        """
        Rotate the SQLCipher encryption key to ``new_key``.

        All existing pooled connections are closed so that subsequent
        operations reopen them with the updated key.
        """
        if not self._cipher_secret or not self._cipher_secret.get_secret_value():
            raise RuntimeError("encryption at rest is not enabled")
        conn = await aiosqlite.connect(self._dsn, uri=True, timeout=5, check_same_thread=False)
        try:
            await conn.execute("PRAGMA key = ?", (self._cipher_secret.get_secret_value(),))
            new_val = new_key.get_secret_value() if isinstance(new_key, SecretStr) else new_key
            await conn.execute("PRAGMA rekey = ?", (new_val,))
            await conn.commit()
        finally:
            with contextlib.suppress(Exception):
                await conn.close()
        self._cipher_secret = SecretStr(new_val)
        while not self._read_pool.empty():
            old = await self._read_pool.get()
            await old.close()
        if self._writer_conn is not None:
            await self._writer_conn.close()
            self._writer_conn = None
        for old in list(self._acquired):
            await old.close()
        self._acquired.clear()
        self._read_created = 0

    # ------------------------------------------------------------------
    # DF/IDF cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> set[str]:
        return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS}

    def _update_df_cache(self, text: str, delta: int) -> None:
        if self._df_cache is None:
            return
        tokens = self._tokenise(text)
        for tok in tokens:
            df, _ = self._df_cache.get(tok) or (0, 0.0)
            df = max(0, df + delta)
            idf = math.log((self._doc_count + 1) / (df + 1)) + 1
            self._df_cache.put(tok, (df, idf))

    def get_token_stats(self, token: str) -> tuple[int, float]:
        """Return (df, idf) for ``token`` from cache or defaults."""
        if self._df_cache is None:
            return (0, 0.0)
        val = self._df_cache.get(token)
        return val if val is not None else (0, 0.0)

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
            self._initialising = asyncio.current_task()
            try:
                try:
                    conn = await self._acquire(write=True)
                except Exception:
                    self._db_ok = False
                    self._write_enabled = False
                    # Propagate initialization failures so callers can handle the
                    # underlying database error rather than hitting a misleading
                    # ``writes disabled`` runtime error on the next operation.
                    raise
                try:
                    await conn.execute(self._CREATE_SQL)
                    await conn.execute(self._CREATE_CASES_SQL)
                    cur = await conn.execute("PRAGMA table_info(memories)")
                    cols = {row[1] for row in await cur.fetchall()}
                    if "memory_type" not in cols:
                        await conn.execute(
                            "ALTER TABLE memories ADD COLUMN memory_type TEXT DEFAULT 'episodic'"
                        )
                    if "ttl_seconds" not in cols:
                        await conn.execute("ALTER TABLE memories ADD COLUMN ttl_seconds INTEGER")
                    if "pinned" not in cols:
                        await conn.execute(
                            "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0"
                        )
                    if "last_used" not in cols:
                        await conn.execute("ALTER TABLE memories ADD COLUMN last_used TEXT")
                        await conn.execute("UPDATE memories SET last_used = created_at")
                    if "success_score" not in cols:
                        await conn.execute(
                            "ALTER TABLE memories ADD COLUMN success_score REAL DEFAULT 0"
                        )
                    if "decay" not in cols:
                        await conn.execute("ALTER TABLE memories ADD COLUMN decay REAL DEFAULT 0")
                    await conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS memory_scores (
                            memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                            score     REAL NOT NULL
                        )
                        """
                    )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_memory_scores_score ON memory_scores(score)"
                    )
                    # Ensure FTS virtual tables and triggers exist
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
                        CREATE VIRTUAL TABLE IF NOT EXISTS cases_fts USING fts5(problem, plan);
                        CREATE TRIGGER IF NOT EXISTS cases_ai AFTER INSERT ON cases BEGIN
                            INSERT INTO cases_fts(rowid, problem, plan) VALUES (new.rowid, new.problem, new.plan);
                        END;
                        CREATE TRIGGER IF NOT EXISTS cases_ad AFTER DELETE ON cases BEGIN
                            DELETE FROM cases_fts WHERE rowid = old.rowid;
                        END;
                        CREATE TRIGGER IF NOT EXISTS cases_au AFTER UPDATE ON cases BEGIN
                            DELETE FROM cases_fts WHERE rowid = old.rowid;
                            INSERT INTO cases_fts(rowid, problem, plan) VALUES (new.rowid, new.problem, new.plan);
                        END;
                        """
                    )
                    # Migration: backfill FTS tables if empty or out of sync
                    cur = await conn.execute("SELECT count(*) FROM memories")
                    mem_count = (await cur.fetchone())[0]
                    cur = await conn.execute("SELECT count(*) FROM memories_fts")
                    fts_count = (await cur.fetchone())[0]
                    if mem_count != fts_count:
                        await conn.execute("DELETE FROM memories_fts")
                        await conn.execute(
                            "INSERT INTO memories_fts(rowid, text) SELECT rowid, text FROM memories"
                        )
                    cur = await conn.execute("SELECT count(*) FROM cases")
                    case_count = (await cur.fetchone())[0]
                    cur = await conn.execute("SELECT count(*) FROM cases_fts")
                    cfts_count = (await cur.fetchone())[0]
                    if case_count != cfts_count:
                        await conn.execute("DELETE FROM cases_fts")
                        await conn.execute(
                            "INSERT INTO cases_fts(rowid, problem, plan) SELECT rowid, problem, plan FROM cases"
                        )
                    # Backwards-compat: ensure newer columns exist for legacy schemas
                    cur = await conn.execute("PRAGMA table_info(memories)")
                    cols = {r[1] for r in await cur.fetchall()}
                    for stmt in [
                        ("level", "INTEGER", "0"),
                        ("episode_id", "TEXT", "NULL"),
                        ("modality", "TEXT", "'text'"),
                        ("connections", "JSON", "NULL"),
                        ("lang", "TEXT", "''"),
                        ("source", "TEXT", "''"),
                        ("content_hash", "TEXT", "''"),
                        ("access_count", "INTEGER", "1"),
                        ("last_access_ts", "REAL", "0"),
                    ]:
                        name, typ, default = stmt
                        if name not in cols:
                            await conn.execute(
                                f"ALTER TABLE memories ADD COLUMN {name} {typ} DEFAULT {default}"
                            )
                    await conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
                    )
                    # Older schemas used in tests may lack some of the newer columns.
                    # Creating indexes conditionally keeps the migration path
                    # tolerant of these legacy layouts.
                    for stmt in [
                        "CREATE INDEX IF NOT EXISTS idx_memories_level ON memories(level)",
                        "CREATE INDEX IF NOT EXISTS idx_memories_episode_id ON memories(episode_id)",
                        "CREATE INDEX IF NOT EXISTS idx_memories_modality ON memories(modality)",
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)",
                    ]:
                        with contextlib.suppress(Exception):
                            await conn.execute(stmt)
                    await conn.commit()
                    if self._df_cache is None:
                        try:
                            from memory_system.settings import get_settings

                            cfg = get_settings()
                            self._df_cache = SmartCache(
                                max_size=cfg.token_cache.size, ttl=cfg.token_cache.ttl_seconds
                            )
                        except Exception:
                            self._df_cache = SmartCache()
                    if self._df_cache is not None:
                        self._df_cache.clear()
                        cur = await conn.execute("SELECT text FROM memories")
                        rows = await cur.fetchall()
                        self._doc_count = len(rows)
                        for (text,) in rows:
                            self._update_df_cache(text, 1)
                    self._initialised = True
                finally:
                    await self._release(conn)
            finally:
                self._initialising = None
            logger.info("SQLiteMemoryStore initialised (dsn=%s)", self._dsn)

    async def aclose(self) -> None:
        """Close all pooled connections and any acquired ones."""
        if self._wal:
            # Ensure the WAL is checkpointed so the -wal file doesn't grow
            # without bound across application restarts.
            with contextlib.suppress(Exception):
                await self._run_wal_checkpoint()
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            loop = asyncio.get_running_loop()
            gather_tasks = [t for t in self._tasks if t.get_loop() is loop]
            if gather_tasks:
                await asyncio.gather(*gather_tasks, return_exceptions=True)
        self._tasks.clear()
        self._wal_checkpoint_task = None
        while not self._read_pool.empty():
            conn = await self._read_pool.get()
            await conn.close()
        if self._writer_conn is not None:
            await self._writer_conn.close()
            self._writer_conn = None
        for conn in list(self._acquired):
            await conn.close()
        self._acquired.clear()
        self._read_created = 0
        self._initialised = False

    async def close(self) -> None:
        """Compatibility alias for ``aclose`` used in tests."""
        await self.aclose()

    def __del__(self):  # pragma: no cover - best effort cleanup
        """
        Ensure background tasks are cancelled when the store is gc'd.

        Some tests intentionally create stores without closing them.  If the
        asynchronous WAL checkpoint task remains referenced by the event loop at
        interpreter shutdown it can produce noisy "Task was destroyed" warnings
        and, in some environments, trigger a hard crash.  Cancelling the task in
        ``__del__`` mitigates this by detaching it from the loop when the store
        object is garbage collected.
        """
        stopper = getattr(self, "_stopping", None)
        if stopper is not None:
            with contextlib.suppress(Exception):
                stopper.set()
        tasks = getattr(self, "_tasks", [])
        for task in tasks:
            with contextlib.suppress(Exception):
                task.cancel()

    # -------------------------------------
    async def add(self, mem: Memory, *, content_hash: str | None = None) -> None:
        """Persist a new :class:`Memory` to the database with upsert semantics."""
        await self.initialise()
        self._ensure_writable()
        if len(mem.text) > MAX_TEXT_LENGTH:
            raise ValueError("text exceeds maximum length")
        meta = mem.metadata or {}
        lang = meta.get("lang") or ""
        source = meta.get("source") or ""
        if content_hash is None:
            normalized = _normalize_for_hash(mem.text)
            payload = f"{normalized}|{lang}|{source}"
            content_hash = blake3_hex(payload.encode())
        event_log = self._event_logger()
        await event_log.log("add", mem.id)
        now_ts = float(meta.get("last_access_ts") or dt.datetime.now(dt.UTC).timestamp())
        raw_access = meta.get("access_count", 1)
        access_count = int(raw_access) if raw_access is not None else 1
        meta["access_count"] = access_count
        meta["last_access_ts"] = now_ts
        async with self.transaction() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM memories WHERE content_hash = ?",
                (content_hash,),
            )
            exists = await cur.fetchone() is not None
            for attempt in range(3):
                try:
                    valid_from = None
                    if (ts := mem.valid_from) is not None:
                        valid_from = ts.isoformat()
                    valid_to = None
                    if (ts := mem.valid_to) is not None:
                        valid_to = ts.isoformat()
                    tx_from = None
                    if (ts := mem.tx_from) is not None:
                        tx_from = ts.isoformat()
                    tx_to = None
                    if (ts := mem.tx_to) is not None:
                        tx_to = ts.isoformat()
                    await conn.execute(
                        "INSERT INTO memories (id, text, created_at, valid_from, valid_to, tx_from, tx_to, importance, valence, emotional_intensity, level, episode_id, modality, connections, metadata, memory_type, lang, source, pinned, ttl_seconds, last_used, success_score, decay, content_hash, access_count, last_access_ts) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(content_hash) DO UPDATE SET "
                        "access_count = access_count + 1, "
                        "last_access_ts = excluded.last_access_ts, "
                        "last_used = excluded.last_used, "
                        "metadata = json_set(COALESCE(metadata, '{}'), '$.access_count', access_count + 1, '$.last_access_ts', excluded.last_access_ts)",
                        (
                            mem.id,
                            mem.text,
                            mem.created_at.isoformat(),
                            valid_from,
                            valid_to,
                            tx_from,
                            tx_to,
                            mem.importance,
                            mem.valence,
                            mem.emotional_intensity,
                            mem.level,
                            mem.episode_id,
                            mem.modality,
                            _safe_json(mem.connections),
                            _safe_json(meta),
                            mem.memory_type,
                            lang,
                            source,
                            int(mem.pinned),
                            mem.ttl_seconds,
                            mem.last_used.isoformat() if mem.last_used else None,
                            mem.success_score,
                            mem.decay,
                            content_hash,
                            access_count,
                            now_ts,
                        ),
                    )
                    if Fernet is not None:
                        cek = Fernet.generate_key()
                        cipher = Fernet(cek).encrypt(mem.text.encode())
                        wrapped = ENVELOPE_MANAGER.encrypt(cek.decode())
                        await conn.execute(
                            "UPDATE memories SET ciphertext = ?, nonce = ?, cek_wrapped = ?, kek_id = ? WHERE id = ?",
                            (cipher, b"", wrapped, "local", mem.id),
                        )
                    break
                except Exception as exc:
                    if "database is locked" in str(exc).lower() and attempt < 2:
                        await asyncio.sleep(0.05)
                        continue
                    raise
        if not exists:
            self._doc_count += 1
            self._update_df_cache(mem.text, 1)
        await self._run_commit_hooks()
        audit_logger.info("memory.add", extra={"id": mem.id})

    async def add_case(self, case: Case) -> None:
        """Persist a new :class:`Case` to the database."""
        await self.initialise()
        self._ensure_writable()
        async with self.transaction() as conn:
            await conn.execute(
                "INSERT INTO cases (id, problem, plan, outcome, evaluation, embedding) VALUES (?, ?, ?, ?, ?, json(?))",
                (
                    case.id,
                    case.problem,
                    case.plan,
                    case.outcome,
                    case.evaluation,
                    json.dumps(case.embedding) if case.embedding is not None else None,
                ),
            )
        await self._run_commit_hooks()
        audit_logger.info("case.add", extra={"id": case.id})

    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None:
        """
        Insert multiple memories using a single transaction.

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
        self._ensure_writable()
        async with self.transaction() as conn:
            await conn.execute("BEGIN")
            sql = (
                "INSERT INTO memories (id, text, created_at, valid_from, valid_to, tx_from, tx_to, importance, valence, emotional_intensity, level, episode_id, modality, connections, metadata, memory_type, lang, source, pinned, ttl_seconds, last_used, success_score, decay, content_hash, access_count, last_access_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(content_hash) DO UPDATE SET "
                "access_count = access_count + 1, "
                "last_access_ts = excluded.last_access_ts, "
                "last_used = excluded.last_used, "
                "metadata = json_set(COALESCE(metadata, '{}'), '$.access_count', access_count + 1, '$.last_access_ts', excluded.last_access_ts)"
            )
            batch: list[tuple[Any, ...]] = []
            for mem in memories:
                if len(mem.text) > MAX_TEXT_LENGTH:
                    raise ValueError("text exceeds maximum length")
                meta = mem.metadata or {}
                lang = meta.get("lang") or ""
                source = meta.get("source") or ""
                normalized = _normalize_for_hash(mem.text)
                chash = blake3_hex(f"{normalized}|{lang}|{source}".encode())
                now_ts = float(meta.get("last_access_ts") or dt.datetime.now(dt.UTC).timestamp())
                raw_acount = meta.get("access_count", 1)
                acount = int(raw_acount) if raw_acount is not None else 1
                meta = dict(meta)
                meta["access_count"] = acount
                meta["last_access_ts"] = now_ts
                valid_from = None
                if (ts := mem.valid_from) is not None:
                    valid_from = ts.isoformat()
                valid_to = None
                if (ts := mem.valid_to) is not None:
                    valid_to = ts.isoformat()
                tx_from = None
                if (ts := mem.tx_from) is not None:
                    tx_from = ts.isoformat()
                tx_to = None
                if (ts := mem.tx_to) is not None:
                    tx_to = ts.isoformat()
                batch.append(
                    (
                        mem.id,
                        mem.text,
                        mem.created_at.isoformat(),
                        valid_from,
                        valid_to,
                        tx_from,
                        tx_to,
                        mem.importance,
                        mem.valence,
                        mem.emotional_intensity,
                        mem.level,
                        mem.episode_id,
                        mem.modality,
                        _safe_json(mem.connections),
                        _safe_json(meta),
                        mem.memory_type,
                        lang,
                        source,
                        int(mem.pinned),
                        mem.ttl_seconds,
                        mem.last_used.isoformat() if mem.last_used else None,
                        mem.success_score,
                        mem.decay,
                        chash,
                        acount,
                        now_ts,
                    )
                )
                if len(batch) >= batch_size:
                    await conn.executemany(sql, batch)
                    batch.clear()
            if batch:
                await conn.executemany(sql, batch)
        self._doc_count += len(memories)
        for m in memories:
            self._update_df_cache(m.text, 1)
        await self._run_commit_hooks()
        self._schedule_wal_checkpoint()
        audit_logger.info("memory.add_many", extra={"ids": [m.id for m in memories]})

    async def add_memories_streaming(
        self,
        iterator: Iterable[dict[str, Any]] | AsyncIterator[dict[str, Any]],
        *,
        batch_size: int = 100,
        save_interval: int | None = None,
    ) -> int:
        """
        Stream memory dictionaries into the store without full buffering.

        Parameters
        ----------
        iterator:
            Source of memory dictionaries. Each item should include at least
            ``text`` and may optionally provide other :class:`Memory` fields.
        batch_size:
            Number of records to accumulate before persisting.
        save_interval:
            Ignored for this backend; accepted for API parity.

        Returns
        -------
        int
            Total number of inserted memories.

        """

        async def _aiter(
            it: Iterable[dict[str, Any]] | AsyncIterator[dict[str, Any]],
        ) -> AsyncIterator[dict[str, Any]]:
            if hasattr(it, "__aiter__"):
                async for item in cast("AsyncIterator[dict[str, Any]]", it):
                    yield item
            else:
                for item in it:
                    yield item

        await self.initialise()
        self._ensure_writable()
        batch: list[Memory] = []
        total = 0
        async for raw in _aiter(iterator):
            batch.append(
                Memory.new(
                    raw["text"],
                    importance=raw.get("importance", 0.0),
                    valence=raw.get("valence", 0.0),
                    emotional_intensity=raw.get("emotional_intensity", 0.0),
                    metadata=raw.get("metadata"),
                    level=raw.get("level", 0),
                    episode_id=raw.get("episode_id"),
                    modality=raw.get("modality", "text"),
                    connections=raw.get("connections"),
                    memory_type=raw.get("memory_type", "episodic"),
                    ttl_seconds=raw.get("ttl_seconds"),
                    last_used=raw.get("last_used"),
                    success_score=raw.get("success_score"),
                    decay=raw.get("decay"),
                )
            )
            if len(batch) >= batch_size:
                await self.add_many(batch)
                total += len(batch)
                batch.clear()

        if batch:
            await self.add_many(batch)
            total += len(batch)

        return total

    async def get(self, memory_id: str) -> Memory | None:
        """Fetch a memory by its ID."""
        await self.initialise()
        conn = await self._acquire()
        try:
            now = dt.datetime.now(dt.UTC).isoformat()
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ? AND valid_from <= ? AND valid_to >= ? AND tx_from <= ? AND tx_to >= ? AND cek_wrapped IS NOT NULL",
                (memory_id, now, now, now, now),
            )
            row = await cursor.fetchone()
            mem = self._row_to_memory(row) if row else None
            audit_logger.info("memory.get", extra={"id": memory_id, "found": mem is not None})
            return mem
        finally:
            await self._release(conn)

    async def ping(self) -> None:
        """Simple connectivity check used by readiness probes."""
        await self.initialise()
        if not self._db_ok:
            raise RuntimeError("database unavailable")
        try:
            conn = await self._acquire()
        except Exception as exc:
            raise RuntimeError("database unavailable") from exc
        try:
            await conn.execute("SELECT 1")
        finally:
            await self._release(conn)

    def _row_to_memory(self, row: aiosqlite.Row | Any) -> Memory:
        """Map a database row or a row-like object to a :class:`Memory`."""

        def _get(obj: Any, key: str) -> Any:
            try:
                return obj[key]
            except (KeyError, TypeError, IndexError):
                # ``sqlite3.Row`` objects raise ``IndexError`` when a column is
                # missing.  Some tests use lightweight stand-ins that may not
                # provide every optional field, so we gracefully fall back to
                # ``getattr`` which returns ``None`` when the attribute doesn't
                # exist.  This mirrors the behaviour of ``dict.get`` and keeps
                # ``_row_to_memory`` robust across different row types.
                return getattr(obj, key, None)

        meta_raw = _get(row, "metadata")
        if meta_raw in (None, "null"):
            metadata = None
        else:
            try:
                metadata = json.loads(meta_raw)
            except json.JSONDecodeError:
                logger.warning("Failed to decode metadata JSON: %s", meta_raw)
                metadata = None

        conn_raw = _get(row, "connections")
        if conn_raw in (None, "null"):
            connections = None
        else:
            try:
                connections = json.loads(conn_raw)
            except json.JSONDecodeError:
                logger.warning("Failed to decode connections JSON: %s", conn_raw)
                connections = None
        level = _get(row, "level")
        modality = _get(row, "modality")
        mtype = _get(row, "memory_type")
        pinned_raw = _get(row, "pinned")
        pinned = bool(pinned_raw) if pinned_raw not in (None, "") else False
        ttl = _get(row, "ttl_seconds")
        last_used_raw = _get(row, "last_used")
        last_used = dt.datetime.fromisoformat(last_used_raw) if last_used_raw else None
        success_score = _get(row, "success_score")
        decay = _get(row, "decay")
        access_count = _get(row, "access_count")
        last_access_ts = _get(row, "last_access_ts")
        created_raw = _get(row, "created_at")
        created_at = dt.datetime.fromisoformat(created_raw)
        vf_raw = _get(row, "valid_from")
        vf = dt.datetime.fromisoformat(vf_raw) if vf_raw else created_at
        vt_raw = _get(row, "valid_to")
        vt = dt.datetime.fromisoformat(vt_raw) if vt_raw else dt.datetime.max.replace(tzinfo=dt.UTC)
        tf_raw = _get(row, "tx_from")
        tf = dt.datetime.fromisoformat(tf_raw) if tf_raw else created_at
        tt_raw = _get(row, "tx_to")
        tt = dt.datetime.fromisoformat(tt_raw) if tt_raw else dt.datetime.max.replace(tzinfo=dt.UTC)

        # Ensure metadata dict and attach derived fields
        meta = metadata or {}
        if access_count is not None:
            meta["access_count"] = access_count
        if last_access_ts is not None:
            with contextlib.suppress(ValueError, TypeError):
                meta["last_access_ts"] = float(last_access_ts)
        meta.setdefault("personalness", 1.0)
        meta.setdefault("globalness", 0.0)

        return Memory(
            id=_get(row, "id"),
            text=_get(row, "text"),
            created_at=created_at,
            valid_from=vf,
            valid_to=vt,
            tx_from=tf,
            tx_to=tt,
            importance=_get(row, "importance"),
            valence=_get(row, "valence"),
            emotional_intensity=_get(row, "emotional_intensity"),
            metadata=meta,
            level=0 if level is None else level,
            episode_id=_get(row, "episode_id"),
            modality="text" if modality is None else modality,
            connections=connections,
            memory_type="episodic" if mtype is None else mtype,
            pinned=pinned,
            ttl_seconds=ttl,
            last_used=last_used,
            success_score=0.0 if success_score is None else success_score,
            decay=0.0 if decay is None else decay,
        )

    def _row_to_case(self, row: aiosqlite.Row | Any) -> Case:
        """Map a database row to a :class:`Case`."""

        def _get(obj: Any, key: str) -> Any:
            try:
                return obj[key]
            except (KeyError, TypeError):
                return getattr(obj, key, None)

        emb_raw = _get(row, "embedding")
        embedding = json.loads(emb_raw) if emb_raw not in (None, "null") else None
        eval_raw = _get(row, "evaluation")
        return Case(
            id=_get(row, "id"),
            problem=_get(row, "problem"),
            plan=_get(row, "plan"),
            outcome=_get(row, "outcome"),
            evaluation=None if eval_raw is None else float(eval_raw),
            embedding=embedding,
        )

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    async def search(
        self,
        text_query: str | None = None,
        *,
        metadata_filters: dict[str, Any] | None = None,
        limit: int = 20,
        level: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Full-text + JSON1 metadata search (no vectors here)."""
        await self.initialise()
        conn = await self._acquire()
        try:
            now = dt.datetime.now(dt.UTC).isoformat()
            params: list[Any] = []
            if text_query:
                sql = (
                    "SELECT m.id, m.text, m.created_at, m.valid_from, m.valid_to, m.tx_from, m.tx_to, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata, "
                    "m.memory_type, m.pinned, m.ttl_seconds, m.last_used, m.success_score, m.decay, "
                    "m.access_count, m.last_access_ts "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.valid_from <= ? AND m.valid_to >= ? AND m.tx_from <= ? AND m.tx_to >= ? AND m.cek_wrapped IS NOT NULL"
                )
                query = text_query if any(ch in text_query for ch in "* ") else f"{text_query}*"
                params.extend([query, now, now, now, now])
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality", "memory_type"}:
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
                clauses: list[str] = [
                    "valid_from <= ?",
                    "valid_to >= ?",
                    "tx_from <= ?",
                    "tx_to >= ?",
                    "cek_wrapped IS NOT NULL",
                ]
                params.extend([now, now, now, now])
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality", "memory_type"}:
                            clauses.append(f"{key} = ?")
                            params.append(val)
                        else:
                            clauses.append("json_extract(metadata, ?) = ?")
                            params.extend([f"$.{key}", val])
                if level is not None:
                    clauses.append("level = ?")
                    params.append(level)
                sql = (
                    "SELECT id, text, created_at, valid_from, valid_to, tx_from, tx_to, importance, valence, emotional_intensity, level, episode_id, modality, "
                    "connections, metadata, memory_type, ttl_seconds, last_used, success_score, decay, access_count, last_access_ts FROM memories"
                )
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
            audit_logger.info(
                "memory.search",
                extra={"query": text_query or "", "returned": len(results)},
            )
            return results
        finally:
            await self._release(conn)

    async def search_iter(
        self,
        text_query: str | None = None,
        *,
        metadata_filters: dict[str, Any] | None = None,
        limit: int | None = None,
        level: int | None = None,
        min_level: int | None = None,
        final: bool | None = None,
        chunk_size: int = 1000,
    ) -> AsyncIterator[list[Memory]]:
        """
        Yield search results in chunks to keep memory usage bounded.

        Results are streamed using a SQLite cursor and ``fetchmany`` to avoid
        loading the entire result set into memory. Additional filters allow
        callers to restrict the range of levels via ``min_level`` and to
        include only memories marked as final or not via ``final``.
        """
        await self.initialise()
        conn = await self._acquire()
        try:
            now = dt.datetime.now(dt.UTC).isoformat()
            params: list[Any] = []
            if text_query:
                sql = (
                    "SELECT m.id, m.text, m.created_at, m.valid_from, m.valid_to, m.tx_from, m.tx_to, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata, "
                    "m.memory_type, m.pinned, m.ttl_seconds, m.last_used, m.success_score, m.decay, "
                    "m.access_count, m.last_access_ts "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.valid_from <= ? AND m.valid_to >= ? AND m.tx_from <= ? AND m.tx_to >= ? AND m.cek_wrapped IS NOT NULL"
                )
                params.extend([text_query, now, now, now, now])
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality", "memory_type"}:
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
                clauses: list[str] = [
                    "valid_from <= ?",
                    "valid_to >= ?",
                    "tx_from <= ?",
                    "tx_to >= ?",
                    "cek_wrapped IS NOT NULL",
                ]
                params.extend([now, now, now, now])
                if metadata_filters:
                    for key, val in metadata_filters.items():
                        if key in {"episode_id", "modality", "memory_type"}:
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
                    "SELECT id, text, created_at, valid_from, valid_to, tx_from, tx_to, importance, valence, emotional_intensity, level, "
                    "episode_id, modality, connections, metadata, memory_type, ttl_seconds, last_used, success_score, decay, access_count, last_access_ts FROM memories"
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
                    rows = await cursor.fetchmany(batch_size)
                    if not rows:
                        break
                    yield [self._row_to_memory(r) for r in rows]
                    fetched += len(rows)
            finally:
                await cursor.close()
        finally:
            await self._release(conn)

    async def list_recent(
        self,
        *,
        n: int = 20,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
    ) -> list[Memory]:
        """Return the most recent *n* memories with optional filters."""
        await self.initialise()
        conn = await self._acquire()
        try:
            now = dt.datetime.now(dt.UTC).isoformat()
            clauses: list[str] = [
                "valid_from <= ?",
                "valid_to >= ?",
                "tx_from <= ?",
                "tx_to >= ?",
                "cek_wrapped IS NOT NULL",
            ]
            params: list[Any] = [now, now, now, now]
            if level is not None:
                clauses.append("level = ?")
                params.append(level)
            if metadata_filter:
                for key, val in metadata_filter.items():
                    if key in {"episode_id", "modality"}:
                        clauses.append(f"{key} = ?")
                        params.append(val)
                    else:
                        clauses.append("json_extract(metadata, ?) = ?")
                        params.extend([f"$.{key}", val])
            sql = (
                "SELECT id, text, created_at, valid_from, valid_to, tx_from, tx_to, importance, valence, emotional_intensity, level, "
                "episode_id, modality, connections, metadata, memory_type, ttl_seconds, last_used, success_score, decay FROM memories"
            )
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(n)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None:
        """
        Insert or update precomputed ranking *scores*.

        Parameters
        ----------
        scores:
            Sequence of ``(memory_id, score)`` pairs.

        """
        if not scores:
            return
        await self.initialise()
        self._ensure_writable()
        async with self.transaction() as conn:
            await conn.execute("BEGIN")
            await conn.executemany(
                "INSERT INTO memory_scores(memory_id, score) VALUES (?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET score = excluded.score",
                scores,
            )
        await self._run_commit_hooks()
        self._schedule_wal_checkpoint()

    async def top_n_by_score(
        self,
        n: int,
        *,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
        weights: ListBestWeights | None = None,
        ids: Sequence[str] | None = None,
    ) -> Sequence[Memory]:
        """
        Return ``n`` memories ordered by ranking score.

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
                now = dt.datetime.now(dt.UTC).isoformat()
                clauses: list[str] = [
                    "m.valid_from <= ?",
                    "m.valid_to >= ?",
                    "m.tx_from <= ?",
                    "m.tx_to >= ?",
                ]
                params: list[Any] = [now, now, now, now]
                if level is not None:
                    clauses.append("m.level = ?")
                    params.append(level)
                if metadata_filter:
                    for key, val in metadata_filter.items():
                        validate_metadata_key(key)
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
                    "SELECT m.id, m.text, m.created_at, m.valid_from, m.valid_to, m.tx_from, m.tx_to, m.importance, m.valence, "
                    "m.emotional_intensity, m.level, m.episode_id, m.modality, m.connections, m.metadata, "
                    "m.memory_type, m.pinned, m.ttl_seconds, m.last_used, m.success_score, m.decay "
                    "FROM memory_scores s JOIN memories m ON m.id = s.memory_id"
                )
                if clauses:
                    sql += " WHERE " + " AND ".join(clauses)
                sql += " ORDER BY s.score DESC LIMIT ?"
                params.append(n)
            else:
                sql, params_seq = build_top_n_by_score_sql(
                    n,
                    weights,
                    level=level,
                    metadata_filter=metadata_filter,
                    ids=ids,
                )
                params = list(params_seq)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return [self._row_to_memory(r) for r in rows]
        finally:
            await self._release(conn)

    async def add_memory(self, memory: Any) -> None:
        """Add a memory object, accepting either :class:`Memory` or a similar object."""
        await self.initialise()
        self._ensure_writable()
        mem_obj = memory
        if isinstance(mem_obj, Memory):
            if mem_obj.ttl_seconds is not None and mem_obj.ttl_seconds < 0:
                raise ValueError("ttl_seconds must be non-negative")
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
                dt.datetime.now(dt.UTC),
            )
            mimportance = getattr(mem_obj, "importance", 0.0)
            mvalence = getattr(mem_obj, "valence", 0.0)
            mintensity = getattr(mem_obj, "emotional_intensity", 0.0)
            mmeta = getattr(mem_obj, "metadata", None) or {}
            mlevel = getattr(mem_obj, "level", 0)
            mepisode = getattr(mem_obj, "episode_id", None)
            mmodality = getattr(mem_obj, "modality", "text")
            mconnections = getattr(mem_obj, "connections", None)
            mmemory_type = getattr(mem_obj, "memory_type", "episodic")
            mttl = getattr(mem_obj, "ttl_seconds", None)
            if mttl is not None and mttl < 0:
                raise ValueError("ttl_seconds must be non-negative")
            mlast_used = getattr(mem_obj, "last_used", dt.datetime.now(dt.UTC))
            msuccess = getattr(mem_obj, "success_score", 0.0)
            mdecay = getattr(mem_obj, "decay", 0.0)
            mpinned = bool(getattr(mem_obj, "pinned", False))
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
                memory_type=mmemory_type,
                pinned=mpinned,
                ttl_seconds=mttl,
                last_used=mlast_used,
                success_score=msuccess,
                decay=mdecay,
            )
        meta = dict(mem_to_add.metadata or {})
        lang = meta.get("lang") or ""
        source = meta.get("source") or ""
        normalized = _normalize_for_hash(mem_to_add.text)
        content_hash = blake3_hex(f"{normalized}|{lang}|{source}".encode())
        now_ts = float(meta.get("last_access_ts") or dt.datetime.now(dt.UTC).timestamp())
        meta.setdefault("access_count", 1)
        meta.setdefault("last_access_ts", now_ts)
        mem_to_add = replace(mem_to_add, metadata=meta)
        await self.add(mem_to_add, content_hash=content_hash)

    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory entry by ID."""
        await self.initialise()
        self._ensure_writable()
        async with self.transaction() as conn:
            cur = await conn.execute("SELECT text FROM memories WHERE id = ?", (memory_id,))
            row = await cur.fetchone()
            text = row[0] if row else None
            await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if text is not None:
            self._doc_count = max(0, self._doc_count - 1)
            self._update_df_cache(text, -1)
        await self._run_commit_hooks()
        audit_logger.info("memory.delete", extra={"id": memory_id})

    async def purge_memory(self, memory_id: str) -> None:
        """Invalidate a memory by removing its wrapped key."""
        await self.initialise()
        self._ensure_writable()
        async with self.transaction() as conn:
            await conn.execute(
                "UPDATE memories SET cek_wrapped = NULL WHERE id = ?",
                (memory_id,),
            )
        await self._run_commit_hooks()
        audit_logger.info("memory.purge", extra={"id": memory_id})

    async def delete(self, memory_id: str) -> None:
        """Alias for :meth:`delete_memory` to satisfy :class:`MetaStore` interface."""
        await self.delete_memory(memory_id)

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
        memory_type: (
            Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] | None
        ) = None,
        pinned: bool | None = None,
        ttl_seconds: int | None = None,
        last_used: dt.datetime | None = None,
        success_score: float | None = None,
        decay: float | None = None,
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
        self._ensure_writable()
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        async with self.transaction() as conn:
            old_text: str | None = None
            now = dt.datetime.now(dt.UTC).isoformat()
            if text is not None:
                cur = await conn.execute("SELECT text FROM memories WHERE id = ?", (memory_id,))
                row = await cur.fetchone()
                old_text = row[0] if row else None
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

            if memory_type is not None:
                await conn.execute(
                    "UPDATE memories SET memory_type = ? WHERE id = ?",
                    (memory_type, memory_id),
                )
            if pinned is not None:
                await conn.execute(
                    "UPDATE memories SET pinned = ? WHERE id = ?",
                    (int(pinned), memory_id),
                )
            if ttl_seconds is not None:
                await conn.execute(
                    "UPDATE memories SET ttl_seconds = ? WHERE id = ?",
                    (ttl_seconds, memory_id),
                )
            if last_used is not None:
                await conn.execute(
                    "UPDATE memories SET last_used = ? WHERE id = ?",
                    (last_used.isoformat(), memory_id),
                )
            if success_score is not None:
                await conn.execute(
                    "UPDATE memories SET success_score = ? WHERE id = ?",
                    (success_score, memory_id),
                )
            if decay is not None:
                await conn.execute(
                    "UPDATE memories SET decay = ? WHERE id = ?",
                    (decay, memory_id),
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

            cursor = await conn.execute(
                "SELECT * FROM memories WHERE id = ? AND valid_from <= ? AND valid_to >= ? AND tx_from <= ? AND tx_to >= ? AND cek_wrapped IS NOT NULL",
                (memory_id, now, now, now, now),
            )
            row = await cursor.fetchone()
        if not row:
            raise RuntimeError("Memory not found")
        if old_text is not None:
            self._update_df_cache(old_text, -1)
            self._update_df_cache(text or "", 1)
        await self._run_commit_hooks()
        audit_logger.info("memory.update", extra={"id": memory_id})
        return self._row_to_memory(row)

    async def update(self, memory_id: str, **kwargs: Any) -> Memory:
        """Alias for :meth:`update_memory` used by :class:`MetaStore`."""
        return await self.update_memory(memory_id, **kwargs)

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter: MutableMapping[str, Any] | None = None,
        level: int | None = None,
        context: MutableMapping[str, Any] | None = None,
    ) -> SearchResults:
        """Search memories by text and optional metadata filters (alias for :meth:`search`)."""
        res = await self.search(
            text_query=query,
            metadata_filters=dict(metadata_filter) if metadata_filter else None,
            limit=k,
            level=level,
        )
        shadow = [m for m in res if (m.metadata or {}).get("shadow")]
        primary = [m for m in res if (m.metadata or {}).get("shadow") is not True]
        return SearchResults(primary, shadow)

    async def search_cases(self, problem: str, k: int = 5) -> list[Case]:
        """Hybrid search over stored cases using BM25 and embeddings."""
        await self.initialise()
        alpha = 0.5
        embed_text: Callable[[str], Sequence[float]] | None
        try:
            from embedder import embed as embed_text
        except ModuleNotFoundError:  # pragma: no cover - embedder optional
            embed_text = None
        qvec: Sequence[float] | None = None
        if embed_text is not None:
            try:
                qvec = embed_text(problem)
            except ModuleNotFoundError:  # numpy missing
                qvec = None

        conn = await self._acquire()
        try:
            params = [
                (problem if any(ch in problem for ch in "* ") else f"{problem}*"),
                k,
            ]
            cur = await conn.execute(
                """
                SELECT c.id, c.problem, c.plan, c.outcome, c.evaluation, c.embedding
                FROM cases_fts JOIN cases c ON c.rowid = cases_fts.rowid
                WHERE cases_fts MATCH ?
                ORDER BY bm25(cases_fts)
                LIMIT ?
                """,
                params,
            )
            rows = await cur.fetchall()
            bm25_cases = [self._row_to_case(r) for r in rows]
            bm25_scores = {c.id: float(k - i) for i, c in enumerate(bm25_cases)}
            case_map: dict[str, Case] = {c.id: c for c in bm25_cases}

            cos_scores: dict[str, float] = {}
            if qvec is not None:
                cur = await conn.execute(
                    "SELECT id, problem, plan, outcome, evaluation, embedding FROM cases"
                )
                all_rows = await cur.fetchall()
                for row in all_rows:
                    c = self._row_to_case(row)
                    if c.embedding is None:
                        continue
                    dot = sum(a * b for a, b in zip(qvec, c.embedding, strict=False))
                    qnorm = math.sqrt(sum(a * a for a in qvec))
                    vnorm = math.sqrt(sum(b * b for b in c.embedding)) or 1.0
                    cos = dot / (qnorm * vnorm) if qnorm and vnorm else 0.0
                    cos_scores[c.id] = cos
                    case_map[c.id] = c
            top_cos = sorted(cos_scores, key=lambda i: cos_scores.get(i, 0.0), reverse=True)[:k]

            def _z(scores: dict[str, float]) -> dict[str, float]:
                if not scores:
                    return {}
                vals = list(scores.values())
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) or 1.0
                return {i: (v - mean) / std for i, v in scores.items()}

            bm25_z = _z(bm25_scores)
            cos_z = _z({i: cos_scores[i] for i in top_cos})
            all_ids = set(bm25_scores) | set(top_cos)
            combined = {
                i: alpha * bm25_z.get(i, 0.0) + (1 - alpha) * cos_z.get(i, 0.0) for i in all_ids
            }
            sorted_ids = sorted(all_ids, key=lambda i: combined[i], reverse=True)[:k]
            return [case_map[i] for i in sorted_ids]
        finally:
            await self._release(conn)


###############################################################################
# FastAPI integration helpers (optional import‑time dep)
###############################################################################


@asynccontextmanager
async def persist_index_on_commit(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
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
async def lifespan_context(app: FastAPI) -> AsyncIterator[None]:  # pragma: no cover
    """FastAPI lifespan function that attaches a SQLiteMemoryStore to ``app.state``."""
    store = SQLiteMemoryStore()
    await store.initialise()
    app.state.memory_store = store
    try:
        yield
    finally:
        await store.aclose()


def get_memory_store(request: Request) -> MemoryStoreProtocol:  # pragma: no cover
    """Return the SQLiteMemoryStore attached to the FastAPI request."""
    return cast("MemoryStoreProtocol", request.app.state.memory_store)


###############################################################################
# Singleton helper
###############################################################################

_STORE: SQLiteMemoryStore | None = None
_STORE_LOCK = asyncio.Lock()


async def get_store(path: str | Path | None = None) -> SQLiteMemoryStore:
    """
    Return process-wide :class:`SQLiteMemoryStore` singleton.

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


if TYPE_CHECKING:  # pragma: no cover - typing only
    from memory_system.core.enhanced_store import EnhancedMemoryStore, HealthComponent
else:
    from importlib import import_module

    class _LazyType(type):
        def __new__(mcls, name, bases=(), attrs=None):
            cls = super().__new__(mcls, name, bases, attrs or {})
            cls._cls = None
            cls._name = name
            return cls

        def _load(cls):
            if cls._cls is None:
                mod = import_module("memory_system.core.enhanced_store")
                cls._cls = getattr(mod, cls._name)
            return cls._cls

        def __call__(cls, *a: Any, **kw: Any) -> Any:
            return cls._load()(*a, **kw)

        def __instancecheck__(cls, instance: Any) -> bool:  # pragma: no cover - proxy
            return isinstance(instance, cls._load())

        def __getattr__(cls, item: str) -> Any:  # pragma: no cover - passthrough
            return getattr(cls._load(), item)

    class EnhancedMemoryStore(metaclass=_LazyType):
        pass

    class HealthComponent(metaclass=_LazyType):
        pass


__all__ = [
    "Case",
    "Memory",
    "SQLiteMemoryStore",
    "get_store",
    "persist_index_on_commit",
]
if EnhancedMemoryStore is not None and HealthComponent is not None:
    __all__.extend(["EnhancedMemoryStore", "HealthComponent"])
