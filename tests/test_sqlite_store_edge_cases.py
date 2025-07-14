import asyncio
from unittest.mock import patch

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_empty_query_returns_all(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    # populate with a few entries
    for i in range(5):
        await store.add(Memory.new(f"text {i}"))

    results = await store.search("")
    assert len(results) == 5


@pytest.mark.asyncio
async def test_none_id_returns_none(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    result = await store.get(None)  # type: ignore[arg-type]
    assert result is None


@pytest.mark.asyncio
async def test_large_volume_add(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    count = 200

    async def add_one(i: int):
        await store.add(Memory.new(f"bulk {i}"))

    await asyncio.gather(*(add_one(i) for i in range(count)))
    results = await store.search("bulk", limit=count)
    assert len(results) == count


@pytest.mark.asyncio
async def test_database_failure(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))

    async def broken_acquire():
        raise OSError("db down")

    with patch.object(store, "_acquire", broken_acquire):
        with pytest.raises(OSError):
            await store.add(Memory.new("x"))
