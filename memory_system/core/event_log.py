from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from dataclasses import dataclass
from typing import cast

from memory_system._vendor import aiosqlite


@dataclass(slots=True)
class Event:
    id: int
    type: str
    payload: str | None
    ts: float
    idempotency_key: str


class EventLog:
    """Lightweight SQLite-backed event log with idempotency keys."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._initialised = False
        self._init_lock = asyncio.Lock()

    async def _connect(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self._dsn, uri=True, timeout=5, check_same_thread=False)
        return cast("aiosqlite.Connection", conn)

    async def initialise(self) -> None:
        if self._initialised:
            return
        async with self._init_lock:
            if self._initialised:
                return
            async with await self._connect() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT NOT NULL,
                        payload TEXT,
                        ts REAL NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        UNIQUE(type, idempotency_key)
                    )
                    """
                )
                await conn.commit()
            self._initialised = True

    async def seen(self, event_type: str, key: str) -> bool:
        await self.initialise()
        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM events WHERE type = ? AND idempotency_key = ?",
                (event_type, key),
            )
            return await cur.fetchone() is not None

    async def log(
        self, event_type: str, payload: str | None, key: str | None = None
    ) -> str:
        """Persist an event returning the ``idempotency_key`` used."""
        await self.initialise()
        ts = dt.datetime.now(dt.UTC).timestamp()
        key = key or uuid.uuid4().hex
        async with await self._connect() as conn:
            await conn.execute(
                "INSERT INTO events(type, payload, ts, idempotency_key) VALUES (?, ?, ?, ?)",
                (event_type, payload, ts, key),
            )
            await conn.commit()
        return key
