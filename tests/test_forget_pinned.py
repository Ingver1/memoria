import datetime as dt

import pytest

np = pytest.importorskip("numpy")

from memory_system.core.maintenance import forget_old_memories
from memory_system.core.store import Memory, SQLiteMemoryStore

try:  # pragma: no cover - optional dependency
    from memory_system.core.index import FaissHNSWIndex
except ImportError:  # pragma: no cover - optional dependency
    FaissHNSWIndex = None  # type: ignore


@pytest.mark.skipif(FaissHNSWIndex is None, reason="faiss not available")
def test_forget_old_memories_skips_pinned(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        index = FaissHNSWIndex(dim=3)
        keep = Memory.new("keep", pinned=True)
        drop = Memory.new("drop")
        await store.add(keep)
        await store.add(drop)
        vec = np.zeros((2, 3), dtype=np.float32)
        index.add_vectors([keep.id, drop.id], vec)
        past = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
        await store.update_memory(keep.id, metadata={"last_accessed": past.isoformat()})
        await store.update_memory(drop.id, metadata={"last_accessed": past.isoformat()})
        deleted = await forget_old_memories(
            store,
            index,
            min_total=0,
            retain_fraction=1.0,
            ttl=60.0,
        )
        assert deleted == 1
        assert await store.get(keep.id) is not None
        assert await store.get(drop.id) is None
        await store.aclose()

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(_inner())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
