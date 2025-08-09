import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore, get_store
from memory_system.unified_memory import add as um_add, reinforce


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    db_file = tmp_path / "mem.db"
    return SQLiteMemoryStore(db_file.as_posix())


async def _count(store: SQLiteMemoryStore) -> int:
    rows = await store.search(limit=1000)
    return len(rows)


def test_memory_new_validation_ok() -> None:
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
def test_memory_new_validation_error(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        Memory.new("x", **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_row_to_memory_roundtrip(store: SQLiteMemoryStore) -> None:
    mem = Memory.new(
        "hello",
        metadata={"foo": 1},
        importance=0.5,
        valence=0.2,
        episode_id="ep1",
        modality="text",
        connections={"bar": 0.1},
    )
    await store.add(mem)

    loaded = await store.get(mem.id)
    assert loaded == mem


@pytest.mark.asyncio
async def test_add_memory_extracts_fields(store: SQLiteMemoryStore) -> None:
    @dataclass
    class ExtMem:
        memory_id: str
        text: str
        episode_id: str | None = None
        modality: str = "text"
        connections: dict[str, float] | None = None
        metadata: dict[str, Any] | None = None

    obj = ExtMem(
        memory_id="m1",
        text="hi",
        episode_id="epX",
        modality="audio",
        connections={"m2": 0.7},
        metadata={"foo": "bar"},
    )
    await store.add_memory(obj)

    loaded = await store.get("m1")
    assert loaded.episode_id == "epX"
    assert loaded.modality == "audio"
    assert loaded.connections == {"m2": 0.7}
    assert loaded.metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_insert_failure_atomic(store: SQLiteMemoryStore) -> None:
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
async def test_concurrent_add_and_search(store: SQLiteMemoryStore) -> None:
    tasks = [store.add(Memory.new(f"t{i}")) for i in range(10)]
    await asyncio.gather(*tasks)

    search_tasks = [store.search("t") for _ in range(5)]
    results = await asyncio.gather(*search_tasks)
    for batch in results:
        assert len(batch) == 10


@pytest.mark.asyncio
async def test_json_error_handling(store: SQLiteMemoryStore) -> None:
    bad = Memory.new("bad", metadata={"a": object()})
    with pytest.raises(TypeError):
        await store.add(bad)

    assert await _count(store) == 0


@pytest.mark.asyncio
async def test_get_store_recreates_on_path_change(tmp_path: Path) -> None:
    """Supplying a different path recreates the singleton store."""
    from memory_system.core import store as store_mod

    # Ensure global store is reset before the test
    if store_mod._STORE is not None:
        await store_mod._STORE.aclose()
        store_mod._STORE = None

    path1 = tmp_path / "one.db"
    store1 = await get_store(path1)

    closed = False
    original_aclose = store1.aclose

    async def wrapped_aclose() -> None:
        nonlocal closed
        closed = True
        await original_aclose()

    store1.aclose = wrapped_aclose  # type: ignore[assignment]

    path2 = tmp_path / "two.db"
    store2 = await get_store(path2)

    assert closed, "Original store should be closed when path changes"
    assert store2 is not store1
    assert store2._path == path2

    await store2.aclose()
    store_mod._STORE = None


def test_update_memory_increments_importance_and_merges_metadata(
    store: SQLiteMemoryStore,
) -> None:
    async def _run() -> None:
        mem = Memory.new("base", importance=0.2, metadata={"a": 1})
        await store.add(mem)

        updated = await store.update_memory(
            mem.id, importance_delta=0.3, metadata={"b": 2}
        )

        assert abs(updated.importance - 0.5) < 1e-6
        assert updated.metadata == {"a": 1, "b": 2}

    asyncio.run(_run())


def test_reinforce_returns_updated_importance(store: SQLiteMemoryStore) -> None:
    async def _run() -> None:
        mem = await um_add("hi", importance=0.1, store=store)
        updated = await reinforce(mem.memory_id, 0.2, store=store)

        assert abs(updated.importance - 0.3) < 1e-6

    asyncio.run(_run())
