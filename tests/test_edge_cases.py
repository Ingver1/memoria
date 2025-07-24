import asyncio
import os
import glob
from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.vector_store import VectorStore


@pytest.mark.asyncio
async def test_sqlite_store_large_volume(tmp_path: 'pytest.TempPathFactory') -> None:
    """Store and retrieve a large number of memories."""
    db_path = os.path.join(str(tmp_path), "large.db")
    store = SQLiteMemoryStore(db_path)

    # Insert many rows concurrently
    async def add_one(i: int) -> None:
        await store.add(Memory.new(f"text {i}"))

    await asyncio.gather(*(add_one(i) for i in range(1000)))

    results = await store.search("text", limit=1000)
    assert len(results) == 1000


@pytest.mark.asyncio
async def test_vector_store_backup(tmp_path: 'pytest.TempPathFactory') -> None:
    """Ensure VectorStore.replicate creates a backup file."""
    vec_path = Path(os.path.join(str(tmp_path), "vectors.index"))
    store = VectorStore(vec_path, dim=16)
    try:
        # Force write a minimal index
        store.add_vector("test", [0.0] * 16)
        await store.flush()
        await store.replicate()
        backups = glob.glob(os.path.join(str(tmp_path), "*.bak"))
        assert backups, "Backup file not created"
    finally:
        store.close()


from typing import Any

def test_api_search_empty_query(test_client: Any) -> None:
    """Search endpoint should reject empty queries."""
    response = test_client.post("/api/v1/memory/search", json={"query": ""})
    assert response.status_code == 422
