import asyncio
from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_wal_checkpoint_truncates(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.db"
    store = SQLiteMemoryStore(
        db_path,
        wal=True,
        wal_interval=10,
        wal_checkpoint_writes=2,
    )
    await store.initialise()
    await store.add(Memory(id="1", text="one"))
    wal_file = db_path.parent / (db_path.name + "-wal")
    assert wal_file.exists()
    assert wal_file.stat().st_size > 0
    await store.add(Memory(id="2", text="two"))
    await asyncio.sleep(0.5)
    assert wal_file.stat().st_size == 0
    await store.aclose()
