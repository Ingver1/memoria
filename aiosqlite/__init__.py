from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Iterable, Sequence, cast


class Row(sqlite3.Row):
    pass

async def connect(dsn: str, uri: bool = False, timeout: float | int = 30) -> "Connection":
    conn = sqlite3.connect(dsn, timeout=timeout, uri=uri, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return Connection(conn)

class Connection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.row_factory = conn.row_factory

    async def execute(self, sql: str, params: Sequence[Any] | Iterable[Any] | tuple[Any, ...] = ()) -> "Cursor":
        return Cursor(self._conn.execute(sql, tuple(params)))

    async def commit(self) -> None:
        self._conn.commit()

    async def rollback(self) -> None:
        self._conn.rollback()
        
    async def close(self) -> None:
        self._conn.close()

    # The Connection object in this stub does not expose fetch methods
    # directly; callers should use the Cursor returned by ``execute``.

class Cursor:
    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur

    async def fetchone(self) -> Row | None:
        return cast(Row | None, self._cur.fetchone())

    async def fetchall(self) -> list[Row]:
        return self._cur.fetchall()
