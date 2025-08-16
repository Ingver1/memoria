import pytest

from memory_system.core.store import SQLiteMemoryStore

pytestmark = pytest.mark.needs_libsql


def test_libsql_disables_wal() -> None:
    """WAL settings are ignored for remote libSQL connections."""
    store = SQLiteMemoryStore("libsql://example.db", wal=True)
    assert store._use_libsql is True
    assert store._wal is False
    store._schedule_wal_checkpoint()
    assert store._wal_checkpoint_task is None
