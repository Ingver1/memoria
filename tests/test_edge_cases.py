import asyncio
import os
from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.vector_store import VectorStore


@pytest.mark.asyncio
async def test_sqlite_store_large_volume(tmp_path):
    """Store and retrieve a large number of memories."""
    db_path = tmp_path / "large.db"
    store = SQLiteMemoryStore(db_path.as_posix())

    # Insert many rows concurrently
    async def add_one(i: int) -> None:
        await store.add(Memory.new(f"text {i}"))

    await asyncio.gather(*(add_one(i) for i in range(1000)))

    results = await store.search("text", limit=1000)
    assert len(results) == 1000


@pytest.mark.asyncio
async def test_vector_store_backup(tmp_path):
    """Ensure VectorStore.replicate creates a backup file."""
    vec_path = tmp_path / "vectors.index"
    store = VectorStore(vec_path, dim=16)
    try:
        # Force write a minimal index
        store.add_vector("test", [0.0] * 16)
        await store.flush()
        await store.replicate()
        backups = list(tmp_path.glob("*.bak"))
        assert backups, "Backup file not created"
    finally:
        store.close()


def test_api_search_empty_query(test_client):
    """Search endpoint should reject empty queries."""
    response = test_client.post("/api/v1/memory/search", json={"query": ""})
    assert response.status_code == 422
