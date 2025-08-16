import sqlite3
import asyncio
from typing import Any
from collections.abc import Sequence

# Expose ``Row`` to mirror the real ``aiosqlite`` API
Row = sqlite3.Row


class Cursor:
    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._cursor = cursor

    async def fetchone(self) -> Any:
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self) -> list[Any]:
        return await asyncio.to_thread(self._cursor.fetchall)

    async def fetchmany(self, size: int) -> list[Any]:
        return await asyncio.to_thread(self._cursor.fetchmany, size)

    async def close(self) -> None:
        await asyncio.to_thread(self._cursor.close)


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

    async def rollback(self) -> None:
        """Rollback the current transaction if supported.

        ``sqlite3`` provides a synchronous ``rollback`` method.  The stubbed
        ``aiosqlite`` connection mirrors this by delegating to ``asyncio``'s
        thread pool.  Some tests expect the method to exist even if no
        transaction is active, so failures are intentionally suppressed.
        """

        try:  # pragma: no cover - error path hard to trigger reliably
            await asyncio.to_thread(self._conn.rollback)
        except Exception:
            pass

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    # ``async with`` support -------------------------------------------------
    async def __aenter__(self) -> "Connection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        await self.close()

class _ConnectContext:
    """Awaitable and async-context-manager wrapper for ``connect``."""

    def __init__(self, coro: "asyncio.Future[Connection]") -> None:
        self._coro = coro
        self._conn: Connection | None = None

    def __await__(self):  # pragma: no cover - exercised indirectly
        return self._coro.__await__()

    async def __aenter__(self) -> Connection:
        self._conn = await self._coro
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self._conn is not None
        await self._conn.close()


def connect(
    dsn: str,
    *,
    uri: bool = False,
    timeout: float | None = None,
    check_same_thread: bool = False,
) -> _ConnectContext:
    """Return an awaitable/context-manager ``Connection`` wrapper."""

    async def _connect() -> Connection:
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

    return _ConnectContext(asyncio.ensure_future(_connect()))
