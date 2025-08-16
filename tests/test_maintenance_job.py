import datetime as dt

import pytest

import memory_system.core.maintenance as mnt
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.utils import metrics  # noqa: F401  # imported for side effects


class DummyIndex:
    def remove_ids(self, ids):
        self.removed = getattr(self, "removed", []) + list(ids)


@pytest.mark.asyncio
async def test_maintenance_run_once_prunes(tmp_path, monkeypatch):
    async def fake_consolidate(store, index, **kwargs):
        return []

    monkeypatch.setattr(mnt, "consolidate_store", fake_consolidate)

    async def fake_forget(store, index, *, ttl=None, **kwargs):
        now = dt.datetime.now(dt.UTC)
        for m in await store.search(limit=100):
            ts = None
            if m.metadata:
                ts = m.metadata.get("last_accessed")
            if (
                ttl is not None
                and ts
                and (now - dt.datetime.fromisoformat(ts)).total_seconds() > ttl
            ):
                await store.delete_memory(m.id)
        return 0

    monkeypatch.setattr(mnt, "forget_old_memories", fake_forget)

    store = SQLiteMemoryStore(tmp_path / "mem.db")
    index = DummyIndex()

    keep = Memory.new("keep")
    drop = Memory.new("drop")
    await store.add_memory(keep)
    await store.add_memory(drop)
    old = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=10)
    await store.update_memory(drop.id, metadata={"last_accessed": old.isoformat()})

    await mnt.run_once(store, index, ttl=1)

    texts = [m.text for m in await store.search(limit=10)]
    assert "drop" not in texts
    assert "keep" in texts
