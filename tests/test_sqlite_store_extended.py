import asyncio
import json

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


class TestMemoryDataclass:
    def test_memory_new_ranges(self):
        mem = Memory.new(
            "hello",
            importance=0.5,
            valence=0.2,
            emotional_intensity=0.7,
            metadata={"foo": "bar"},
        )
        assert 0.0 <= mem.importance <= 1.0
        assert -1.0 <= mem.valence <= 1.0
        assert 0.0 <= mem.emotional_intensity <= 1.0
        assert mem.metadata == {"foo": "bar"}
        assert mem.created_at.tzinfo is not None

    def test_row_to_memory_serialization(self):
        mem = Memory.new("serialize", metadata={"a": 1})
        store = SQLiteMemoryStore(":memory:")
        row = {
            "id": mem.id,
            "text": mem.text,
            "created_at": mem.created_at.isoformat(),
            "importance": mem.importance,
            "valence": mem.valence,
            "emotional_intensity": mem.emotional_intensity,
            "metadata": json.dumps(mem.metadata),
        }
        row_obj = type("Row", (), row)
        restored = store._row_to_memory(row_obj)
        assert restored == mem


@pytest.mark.asyncio
async def test_add_transaction_atomicity(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    bad = Memory.new("bad", metadata={"a": object()})
    size_before = store._pool.qsize()
    with pytest.raises(TypeError):
        await store.add(bad)
    assert store._pool.qsize() == size_before


@pytest.mark.asyncio
async def test_concurrent_add_and_search(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))

    async def add_one(i: int):
        await store.add(Memory.new(f"text {i}"))

    await asyncio.gather(*(add_one(i) for i in range(10)))
    results = await store.search("text")
    assert len(results) == 10


@pytest.mark.asyncio
async def test_search_no_results(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    results = await store.search("nothing")
    assert results == []
