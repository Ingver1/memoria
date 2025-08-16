import datetime as dt
from pathlib import Path

import pytest

import memory_system.unified_memory as um
from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.unified_memory import add as um_add, reinforce


@pytest.mark.asyncio
async def test_reinforce_is_monotonic(tmp_path: Path) -> None:
    db_path = tmp_path / "reinforce.db"
    store = SQLiteMemoryStore(db_path)
    await store.initialise()

    mem_small = await um_add("m1", importance=0.1, store=store)
    mem_large = await um_add("m2", importance=0.1, store=store)

    updated_small = await reinforce(mem_small.memory_id, 0.1, store=store)
    updated_large = await reinforce(mem_large.memory_id, 0.2, store=store)

    assert updated_large.importance > updated_small.importance

    await store.aclose()


def test_decay_is_monotonic() -> None:
    now = dt.datetime.now(dt.UTC)
    recent = now - dt.timedelta(days=1)
    older = now - dt.timedelta(days=5)

    mem_recent = Memory(
        id="recent",
        text="r",
        created_at=now,
        importance=0.0,
        valence=0.0,
        emotional_intensity=1.0,
        metadata={"last_accessed": recent.isoformat()},
    )
    mem_older = Memory(
        id="older",
        text="o",
        created_at=now,
        importance=0.0,
        valence=0.0,
        emotional_intensity=1.0,
        metadata={"last_accessed": older.isoformat()},
    )

    dyn = MemoryDynamics()
    recent_score = dyn.score(mem_recent, now=now)
    older_score = dyn.score(mem_older, now=now)

    assert older_score < recent_score


@pytest.mark.asyncio
async def test_level_and_metadata_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "filters.db"
    store = SQLiteMemoryStore(db_path)
    await store.initialise()

    mem0 = Memory(id="m0", text="base", level=0, metadata={"user_id": "u1"})
    mem1 = Memory(id="m1", text="lvl1", level=1, metadata={"user_id": "u2"})
    await store.add_memory(mem0)
    await store.add_memory(mem1)
    await store.upsert_scores([(mem0.id, 0.1), (mem1.id, 0.2)])

    best_lvl1 = await um.list_best(n=5, store=store, level=1)
    assert [m.memory_id for m in best_lvl1] == [mem1.id]

    best_user = await um.list_best(n=5, store=store, metadata_filter={"user_id": "u1"})
    assert [m.memory_id for m in best_user] == [mem0.id]

    await store.aclose()
