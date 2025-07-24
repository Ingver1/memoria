from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Iterable, Sequence, cast


class Row(sqlite3.Row):
    """Row type for async sqlite stub."""
    pass

async def connect(dsn: str, uri: bool = False, timeout: float | int = 30) -> "Connection":
    """Async connect to sqlite DB."""
    conn = sqlite3.connect(dsn, timeout=timeout, uri=uri, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return Connection(conn)

class Connection:
    """Async wrapper for sqlite3.Connection."""
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.row_factory = conn.row_factory

    async def execute(self, sql: str, params: Sequence[Any] | Iterable[Any] | tuple[Any, ...] = ()) -> "Cursor":
        return Cursor(self._conn.execute(sql, tuple(params)))

    async def executemany(self, sql: str, seq_of_params: Iterable[Sequence[Any]]) -> "Cursor":
        return Cursor(self._conn.executemany(sql, seq_of_params))

    async def executescript(self, script: str) -> "Cursor":
        return Cursor(self._conn.executescript(script))

    async def commit(self) -> None:
        self._conn.commit()

    async def rollback(self) -> None:
        self._conn.rollback()

    async def close(self) -> None:
        self._conn.close()

    async def __aenter__(self) -> "Connection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # The Connection object in this stub does not expose fetch methods
    # directly; callers should use the Cursor returned by ``execute``.

class Cursor:
    """Async wrapper for sqlite3.Cursor."""
    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur

    async def fetchone(self) -> Row | None:
        return cast(Row | None, self._cur.fetchone())

    async def fetchall(self) -> list[Row]:
        return self._cur.fetchall()

    @property
    def description(self) -> Any:
        return self._cur.description

    @property
    def lastrowid(self) -> Any:
        return self._cur.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    async def __aenter__(self) -> "Cursor":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        pass
