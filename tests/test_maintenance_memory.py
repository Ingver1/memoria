import tracemalloc

import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)

from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.store import Memory


@pytest.mark.asyncio
async def test_consolidate_store_memory_usage_bounded(store, index, fake_embed, monkeypatch):
    big_text = "cat " + "x" * 9_996
    for _ in range(1_000):
        mem = Memory.new(big_text)
        await store.add(mem)
        vec = fake_embed(mem.text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([mem.id], vec.astype(np.float32))

    import memory_system.core.maintenance as maint

    monkeypatch.setattr(maint, "embed_text", fake_embed)

    tracemalloc.start()
    await consolidate_store(store, index, threshold=0.9, max_fetch=5_000, chunk_size=100)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 12 * 1024 * 1024

    remaining = await store.search(limit=5_000)
    assert index.stats().total_vectors == len(remaining)


@pytest.mark.asyncio
async def test_forget_old_memories_memory_usage_bounded(store, index, fake_embed):
    big_text = "cat " + "x" * 9_996
    for i in range(1_000):
        mem = Memory.new(big_text, importance=float(i % 100) / 100.0)
        await store.add(mem)
        vec = fake_embed(mem.text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([mem.id], vec.astype(np.float32))

    tracemalloc.start()
    deleted = await forget_old_memories(
        store,
        index,
        min_total=500,
        retain_fraction=0.5,
        max_fetch=5_000,
        chunk_size=100,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 12 * 1024 * 1024

    remaining = await store.search(limit=5_000)
    assert index.stats().total_vectors == len(remaining)
    assert deleted == 1_000 - len(remaining)
