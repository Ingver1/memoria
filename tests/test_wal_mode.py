import pytest

from memory_system.core.store import SQLiteMemoryStore


@pytest.mark.asyncio
async def test_wal_enabled_and_checkpoint_task(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    # Ensure journal mode is WAL
    conn = await store._acquire()
    try:
        cur = await conn.execute("PRAGMA journal_mode")
        assert (await cur.fetchone())[0] == "wal"
    finally:
        await store._release(conn)

    # Check that WAL checkpoint task is scheduled
    assert store._wal_checkpoint_task is not None
    await store.aclose()


def test_vector_store_wal_mode(tmp_path):
    from memory_system.core.vector_store import VectorStore

    with VectorStore(tmp_path / "vec", dim=3) as store:
        cur = store._conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0] == "wal"
