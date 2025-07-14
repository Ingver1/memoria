import asyncio
import tempfile
from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    db_file = tmp_path / "mem.db"
    return SQLiteMemoryStore(db_file.as_posix())


async def _count(store: SQLiteMemoryStore) -> int:
    rows = await store.search(limit=1000)
    return len(rows)


def test_memory_new_validation_ok():
    mem = Memory.new(
        "hi",
        importance=1.0,
        valence=-1.0,
        emotional_intensity=0.0,
    )
    assert mem.importance == 1.0
    assert mem.valence == -1.0
    assert mem.emotional_intensity == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("importance", 1.1),
        ("importance", -0.1),
        ("valence", -1.1),
        ("valence", 1.1),
        ("emotional_intensity", 1.1),
        ("emotional_intensity", -0.1),
    ],
)
def test_memory_new_validation_error(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        Memory.new("x", **kwargs)


@pytest.mark.asyncio
async def test_row_to_memory_roundtrip(store):
    mem = Memory.new("hello", metadata={"foo": 1}, importance=0.5, valence=0.2)
    await store.add(mem)

    loaded = await store.get(mem.id)
    assert loaded == mem


@pytest.mark.asyncio
async def test_insert_failure_atomic(store):
    mem1 = Memory.new("one")
    await store.add(mem1)

    dup = Memory(mem1.id, "dupe", mem1.created_at)
    with pytest.raises(Exception):
        await store.add(dup)

    assert await _count(store) == 1

    mem2 = Memory.new("two")
    await store.add(mem2)
    assert await _count(store) == 2


@pytest.mark.asyncio
async def test_concurrent_add_and_search(store):
    tasks = [store.add(Memory.new(f"t{i}")) for i in range(10)]
    await asyncio.gather(*tasks)

    search_tasks = [store.search("t") for _ in range(5)]
    results = await asyncio.gather(*search_tasks)
    for batch in results:
        assert len(batch) == 10


@pytest.mark.asyncio
async def test_json_error_handling(store):
    bad = Memory.new("bad", metadata={"a": object()})
    with pytest.raises(TypeError):
        await store.add(bad)

    assert await _count(store) == 0
