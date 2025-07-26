"""
Ensures DB columns obey NOT NULL + dimension constraints after inserts.
Works with SQLite or SQLCipher.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim


@pytest.mark.asyncio
async def test_db_invariants(tmp_path: Path) -> None:
    """Test database column invariants after inserts."""
    cfg = UnifiedSettings.for_testing()
    # Use database.db_path instead of storage.database_url
    cfg.database.db_path = tmp_path / "inv.db"
    store = EnhancedMemoryStore(cfg)

    await store.add_memory(text="inv", embedding=np.random.rand(DIM).tolist())

    # direct SQL query for invariant check
    conn = await store._store._acquire()
    try:
        cursor = await conn.execute("SELECT text FROM memories")
        rows = await cursor.fetchall()
    finally:
        await store._store._release(conn)
    text = rows[0]["text"]

    # column text must never be NULL
    assert text is not None

    # vector index dimension must match config
    assert store._index.stats().dim == DIM
