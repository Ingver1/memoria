import sqlite3
import asyncio
from typing import Any, Sequence

Row = sqlite3.Row


class Cursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> Any:
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self) -> list[Any]:
        return await asyncio.to_thread(self._cursor.fetchall)


class Connection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._conn.row_factory = value

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> Cursor:
        if params is None:
            params = ()
        cur = await asyncio.to_thread(self._conn.execute, sql, params)
        return Cursor(cur)

    async def executemany ( self, sql: str , seq_of_params: Sequence [ Sequence [ Any ]] ) -> None :
        await asyncio.to_thread( self ._conn.executemany, sql, seq_of_params)

    async def executescript(self, script: str) -> None:
        await asyncio.to_thread(self._conn.executescript, script)

    async def commit(self) -> None:
        await asyncio.to_thread(self._conn.commit)

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

async def connect(
    dsn: str,
    *,
    uri: bool = False,
    timeout: float | None = None,
    check_same_thread: bool = True,
) -> Connection:
    """
    Lightweight async wrapper around `sqlite3.connect`.
    We expose `timeout` for compatibility with the real aiosqlite API,
    but keep it optional to preserve existing call sites.
    """
    if timeout is None:
        conn = await asyncio.to_thread(
            sqlite3.connect, dsn, uri=uri, check_same_thread=check_same_thread
        )
    else:
        conn = await asyncio.to_thread(
            sqlite3.connect,
            dsn,
            uri=uri,
            timeout=timeout,
            check_same_thread=check_same_thread,
        )
    conn.row_factory = sqlite3.Row
    return Connection(conn)
