import pytest

from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore, persist_index_on_commit

np = pytest.importorskip("numpy")


@pytest.mark.asyncio
async def test_index_saved_automatically(tmp_path):
    db_path = tmp_path / "db.sqlite"
    vec_path = tmp_path / "index.faiss"
    store = SQLiteMemoryStore(db_path.as_posix())
    await store.initialise()
    index = FaissHNSWIndex(dim=3)

    async with persist_index_on_commit(store, index, vec_path.as_posix()):
        mem1 = Memory.new("first")
        index.add_vectors([mem1.id], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        await store.add(mem1)

    assert vec_path.exists()
    new_index = FaissHNSWIndex(dim=3)
    new_index.load(vec_path.as_posix())
    assert new_index.stats().total_vectors >= 1
