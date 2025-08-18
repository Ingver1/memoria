import datetime as dt
import pytest

np = pytest.importorskip("numpy")

try:  # pragma: no cover - optional dependency
    from memory_system.core.index import FaissHNSWIndex
except ImportError:  # pragma: no cover - optional dependency
    FaissHNSWIndex = None  # type: ignore

from memory_system.core.maintenance import forget_old_memories
from memory_system.core.store import Memory, SQLiteMemoryStore


@pytest.mark.skipif(FaissHNSWIndex is None, reason="faiss not available")
def test_forget_old_memories_respects_ttl(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        index = FaissHNSWIndex(dim=3)
        mem = Memory.new("old")
        await store.add(mem)
        vec = np.zeros((1, 3), dtype=np.float32)
        index.add_vectors([mem.id], vec)
        past = dt.datetime.now(dt.UTC) - dt.timedelta(days=2)
        await store.update_memory(mem.id, metadata={"last_accessed": past.isoformat()})
        deleted = await forget_old_memories(
            store,
            index,
            min_total=0,
            retain_fraction=1.0,
            ttl=60.0,
        )
        assert deleted == 1
        assert await store.get(mem.id) is None
        await store.aclose()

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(_inner())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")


@pytest.mark.skipif(FaissHNSWIndex is None, reason="faiss not available")
def test_forget_old_memories_keeps_recent(tmp_path) -> None:
    async def _inner() -> None:
        store = SQLiteMemoryStore(tmp_path / "mem.db")
        await store.initialise()
        index = FaissHNSWIndex(dim=3)
        mem = Memory.new("fresh")
        await store.add(mem)
        vec = np.zeros((1, 3), dtype=np.float32)
        index.add_vectors([mem.id], vec)
        await store.update_memory(
            mem.id, metadata={"last_accessed": dt.datetime.now(dt.UTC).isoformat()}
        )
        deleted = await forget_old_memories(
            store,
            index,
            min_total=0,
            retain_fraction=1.0,
            ttl=60.0,
        )
        assert deleted == 0
        assert await store.get(mem.id) is not None
        await store.aclose()

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(_inner())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
