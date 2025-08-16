from pathlib import Path

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


def _count_db_fds(db_path: Path) -> int:
    base = str(db_path)
    fd_dir = Path("/proc/self/fd")
    count = 0
    for fd in fd_dir.iterdir():
        try:
            target = Path(fd).readlink()
        except OSError:
            continue
        if target.startswith(base):
            count += 1
    return count


@pytest.mark.asyncio
async def test_aclose_closes_unreleased_connection(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    store = SQLiteMemoryStore(db_path.as_posix())
    await store._acquire()
    assert _count_db_fds(db_path) > 0
    await store.aclose()
    assert _count_db_fds(db_path) == 0


@pytest.mark.asyncio
async def test_aclose_after_usage_leaves_no_fds(tmp_path: Path) -> None:
    db_path = tmp_path / "db.sqlite"
    store = SQLiteMemoryStore(db_path.as_posix())
    await store.add(Memory.new("x"))
    assert _count_db_fds(db_path) > 0
    await store.aclose()
    assert _count_db_fds(db_path) == 0
