import asyncio
import json
import logging
from pathlib import Path

import pytest

from memory_system._vendor import aiosqlite
from memory_system.core.store import Memory, SQLiteMemoryStore


class TestMemoryDataclass:
    def test_memory_new_ranges(self) -> None:
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
        assert mem.metadata == {"foo": "bar", "trust_score": 1.0, "error_flag": False}
        assert mem.created_at.tzinfo is not None

    def test_row_to_memory_serialization(self) -> None:
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
        assert restored.id == mem.id
        assert restored.text == mem.text
        assert restored.created_at == mem.created_at
        assert restored.metadata is not None
        assert restored.metadata.get("a") == 1

    def test_row_to_memory_invalid_metadata(self, caplog: pytest.LogCaptureFixture) -> None:
        mem = Memory.new("serialize", connections={"a": 1.0})
        store = SQLiteMemoryStore(":memory:")
        row = {
            "id": mem.id,
            "text": mem.text,
            "created_at": mem.created_at.isoformat(),
            "importance": mem.importance,
            "valence": mem.valence,
            "emotional_intensity": mem.emotional_intensity,
            "metadata": "{bad",  # malformed JSON
            "connections": json.dumps(mem.connections),
        }
        row_obj = type("Row", (), row)
        with caplog.at_level(logging.WARNING):
            restored = store._row_to_memory(row_obj)
        assert restored.metadata is not None
        assert restored.metadata.get("trust_score") == 1.0
        assert restored.metadata.get("error_flag") is False
        assert "Failed to decode metadata JSON" in caplog.text

    def test_row_to_memory_invalid_connections(self, caplog: pytest.LogCaptureFixture) -> None:
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
            "connections": "{bad",  # malformed JSON
        }
        row_obj = type("Row", (), row)
        with caplog.at_level(logging.WARNING):
            restored = store._row_to_memory(row_obj)
        assert restored.connections is None
        assert "Failed to decode connections JSON" in caplog.text


@pytest.mark.asyncio
async def test_add_transaction_atomicity(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    bad = Memory.new("bad", metadata={"a": object()})
    size_before = store._pool.qsize()
    with pytest.raises(TypeError):
        await store.add(bad)
    assert store._pool.qsize() == size_before


@pytest.mark.asyncio
async def test_concurrent_add_and_search(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))

    async def add_one(i: int) -> None:
        await store.add(Memory.new(f"text {i}"))

    await asyncio.gather(*(add_one(i) for i in range(10)))
    results = await store.search("text")
    assert len(results) == 10


@pytest.mark.asyncio
async def test_search_no_results(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    results = await store.search("nothing")
    assert results == []


@pytest.mark.asyncio
async def test_fts_prefix_query(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    await store.add(Memory.new("prefixmatch"))
    results = await store.search("pref*")
    assert len(results) == 1
    assert results[0].text == "prefixmatch"


@pytest.mark.asyncio
async def test_migration_creates_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    # create legacy schema without FTS
    async with aiosqlite.connect(db_path.as_posix()) as conn:
        await conn.execute(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                importance REAL DEFAULT 0,
                valence REAL DEFAULT 0,
                emotional_intensity REAL DEFAULT 0,
                metadata JSON
            );
            """
        )
        mem = Memory.new("legacy text")
        await conn.execute(
            "INSERT INTO memories (id, text, created_at, importance, valence, emotional_intensity, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, json(?))",
            (
                mem.id,
                mem.text,
                mem.created_at.isoformat(),
                mem.importance,
                mem.valence,
                mem.emotional_intensity,
                json.dumps(mem.metadata) if mem.metadata else "null",
            ),
        )
        await conn.commit()

    store = SQLiteMemoryStore(db_path.as_posix())
    results = await store.search("legacy")
    assert results and results[0].text == "legacy text"

    # ensure triggers keep FTS in sync
    await store.add(Memory.new("new memory"))
    res_new = await store.search("new")
    assert any(r.text == "new memory" for r in res_new)

    async with aiosqlite.connect(db_path.as_posix()) as conn:
        cur = await conn.execute("SELECT count(*) FROM memories")
        mem_count = (await cur.fetchone())[0]
        cur = await conn.execute("SELECT count(*) FROM memories_fts")
        fts_count = (await cur.fetchone())[0]
        assert fts_count == mem_count
