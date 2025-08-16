"""
Ensures DB columns obey NOT NULL + dimension constraints after inserts.
Works with SQLite or SQLCipher.
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import DatabaseConfig, UnifiedSettings

DIM = UnifiedSettings.for_testing().model.vector_dim


@pytest.mark.asyncio
async def test_db_invariants(tmp_path: Path) -> None:
    """Test database column invariants after inserts."""
    cfg = UnifiedSettings.for_testing()
    # Replace database configuration with a new path
    cfg.database = DatabaseConfig(db_path=tmp_path / "inv.db")
    store = EnhancedMemoryStore(cfg)
    await store.start()

    await store.add_memory(text="inv", embedding=np.random.rand(DIM).tolist())

    # direct SQL query for invariant check
    conn = await store._store._acquire()
    try:
        cursor = await conn.execute("SELECT text FROM memories")
        rows = await cursor.fetchall()
    finally:
        await store._store._release(conn)
    text = rows[0]["text"]

    # ensure index on created_at exists
    conn = await store._store._acquire()
    try:
        cursor = await conn.execute("PRAGMA index_list('memories')")
        indexes = await cursor.fetchall()
    finally:
        await store._store._release(conn)
    index_names = {row["name"] for row in indexes}
    assert "idx_memories_created_at" in index_names

    # column text must never be NULL
    assert text is not None

    # vector index dimension must match config
    assert store._index.stats().dim == DIM
    await store.close()
