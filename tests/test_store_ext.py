import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skip("requires stable async runtime")

from memory_system._vendor import aiosqlite
from memory_system.core.store import Memory, SQLiteMemoryStore, get_store
from memory_system.unified_memory import add as um_add, reinforce, update as um_update


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
    "field,value,expected",
    [
        ("importance", 1.1, 1.0),
        ("importance", -0.1, 0.0),
        ("valence", -1.1, -1.0),
        ("valence", 1.1, 1.0),
        ("emotional_intensity", 1.1, 1.0),
        ("emotional_intensity", -0.1, 0.0),
    ],
)
def test_memory_new_clamps(field: str, value: float, expected: float) -> None:
    kwargs = {field: value}
    mem = Memory.new("x", **kwargs)  # type: ignore[arg-type]
    assert getattr(mem, field) == pytest.approx(expected)


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
    assert loaded.metadata == {"foo": "bar", "trust_score": 1.0, "error_flag": False}


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


@pytest.mark.asyncio
async def test_update_memory_increments_importance_and_merges_metadata(
    store: SQLiteMemoryStore,
) -> None:
    mem = Memory.new("base", importance=0.2, metadata={"a": 1})
    await store.add(mem)
    updated = await store.update_memory(mem.id, importance_delta=0.3, metadata={"b": 2})
    assert abs(updated.importance - 0.5) < 1e-6
    assert updated.metadata == {"a": 1, "b": 2, "trust_score": 1.0, "error_flag": False}


@pytest.mark.asyncio
async def test_update_memory_clamps_importance_upper_bound(store: SQLiteMemoryStore) -> None:
    mem = Memory.new("base", importance=0.9)
    await store.add(mem)
    updated = await store.update_memory(mem.id, importance_delta=0.5)
    assert abs(updated.importance - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_update_memory_clamps_importance_lower_bound(store: SQLiteMemoryStore) -> None:
    mem = Memory.new("base", importance=0.1)
    await store.add(mem)
    updated = await store.update_memory(mem.id, importance_delta=-0.5)
    assert abs(updated.importance - 0.0) < 1e-6


@pytest.mark.asyncio
async def test_reinforce_returns_updated_importance(store: SQLiteMemoryStore) -> None:
    mem = await um_add("hi", importance=0.1, valence=0.2, emotional_intensity=0.4, store=store)
    updated = await reinforce(mem.memory_id, 0.2, store=store)
    assert abs(updated.importance - 0.3) < 1e-6
    assert updated.valence == pytest.approx(0.2)
    assert updated.emotional_intensity == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_reinforce_combined_fields(store: SQLiteMemoryStore) -> None:
    mem = await um_add(
        "hi",
        importance=0.1,
        valence=0.0,
        emotional_intensity=0.2,
        store=store,
    )
    updated = await reinforce(
        mem.memory_id,
        0.2,
        valence_delta=0.3,
        intensity_delta=0.4,
        store=store,
    )
    assert updated.importance == pytest.approx(0.3)
    assert updated.valence == pytest.approx(0.3)
    assert updated.emotional_intensity == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_update_memory_valence_delta_clamps(store: SQLiteMemoryStore) -> None:
    mem = Memory.new("base", valence=0.0)
    await store.add(mem)

    updated = await store.update_memory(mem.id, valence_delta=0.5)
    assert abs(updated.valence - 0.5) < 1e-6

    updated = await store.update_memory(mem.id, valence_delta=1.0)
    assert abs(updated.valence - 1.0) < 1e-6

    updated = await store.update_memory(mem.id, valence_delta=-3.0)
    assert abs(updated.valence - (-1.0)) < 1e-6


@pytest.mark.asyncio
async def test_update_memory_emotional_intensity_delta_clamps(
    store: SQLiteMemoryStore,
) -> None:
    mem = Memory.new("base", emotional_intensity=0.4)
    await store.add(mem)

    updated = await store.update_memory(mem.id, emotional_intensity_delta=0.3)
    assert abs(updated.emotional_intensity - 0.7) < 1e-6

    updated = await store.update_memory(mem.id, emotional_intensity_delta=1.0)
    assert abs(updated.emotional_intensity - 1.0) < 1e-6

    updated = await store.update_memory(mem.id, emotional_intensity_delta=-2.0)
    assert abs(updated.emotional_intensity - 0.0) < 1e-6


@pytest.mark.asyncio
async def test_unified_update_applies_deltas(store: SQLiteMemoryStore) -> None:
    mem = await um_add("hi", importance=0.1, store=store)
    updated = await um_update(
        mem.memory_id,
        valence_delta=0.6,
        emotional_intensity_delta=0.4,
        importance_delta=0.2,
        store=store,
    )
    assert abs(updated.valence - 0.6) < 1e-6
    assert abs(updated.emotional_intensity - 0.4) < 1e-6
    assert abs(updated.importance - 0.3) < 1e-6


@pytest.mark.asyncio
async def test_unified_update_sets_importance(store: SQLiteMemoryStore) -> None:
    mem = await um_add("hi", importance=0.1, store=store)
    updated = await um_update(mem.memory_id, importance=0.8, store=store)
    assert abs(updated.importance - 0.8) < 1e-6


@pytest.mark.asyncio
async def test_unified_update_sets_emotions(store: SQLiteMemoryStore) -> None:
    async def noop(_: list[tuple[str, float]]) -> None:
        return None

    store.upsert_scores = noop  # type: ignore[assignment]

    mem = Memory.new("hi", valence=0.1, emotional_intensity=0.2)
    await store.add(mem)
    updated = await um_update(
        mem.id,
        valence=-0.5,
        emotional_intensity=0.9,
        store=store,
    )
    assert updated.valence == pytest.approx(-0.5)
    assert updated.emotional_intensity == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_list_recent_metadata_filter(store: SQLiteMemoryStore) -> None:
    m1 = Memory.new("one", episode_id="ep1", modality="text")
    m2 = Memory.new("two", episode_id="ep2", modality="image")
    await store.add(m1)
    await store.add(m2)

    res_ep1 = await store.list_recent(metadata_filter={"episode_id": "ep1"})
    assert [m.id for m in res_ep1] == [m1.id]

    res_mod = await store.list_recent(metadata_filter={"modality": "image"})
    assert [m.id for m in res_mod] == [m2.id]


@pytest.mark.asyncio
async def test_schema_has_episode_modality_indexes(store: SQLiteMemoryStore) -> None:
    await store.initialise()
    conn = await aiosqlite.connect(store._path.as_posix())  # type: ignore[attr-defined]
    try:
        cur = await conn.execute("PRAGMA table_info(memories)")
        cols = {row[1] for row in await cur.fetchall()}
        assert {"episode_id", "modality"}.issubset(cols)
        cur = await conn.execute("PRAGMA index_list(memories)")
        idx = {row[1] for row in await cur.fetchall()}
        assert "idx_memories_episode_id" in idx
        assert "idx_memories_modality" in idx
    finally:
        await conn.close()
