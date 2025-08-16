import pytest

from memory_system.core.store import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_pragmas_applied(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    conn = await store._acquire(write=True)
    try:
        cur = await conn.execute("PRAGMA journal_mode")
        assert (await cur.fetchone())[0] == "wal"
        cur = await conn.execute("PRAGMA synchronous")
        assert (await cur.fetchone())[0] == 1  # NORMAL
        cur = await conn.execute("PRAGMA foreign_keys")
        assert (await cur.fetchone())[0] == 1
    finally:
        await store._release(conn)
        await store.aclose()


@pytest.mark.asyncio
async def test_read_only_skips_pragmas(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    # Alter settings so a read-only connection would attempt to change the
    # synchronous mode if PRAGMAs were executed.
    store._wal = False
    store._synchronous = "EXTRA"
    conn = await store._get_connection(read_only=True)
    try:
        cur = await conn.execute("PRAGMA synchronous")
        # Default FULL (2) remains, confirming no PRAGMA was run
        assert (await cur.fetchone())[0] == 2
    finally:
        await conn.close()
        await store.aclose()
