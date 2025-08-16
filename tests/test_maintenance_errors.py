import logging

import pytest

from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.store import Memory

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)


pytestmark = pytest.mark.asyncio


async def _add_with_vector(store, index, mem, fake_embed):
    await store.add(mem)
    vec = fake_embed(mem.text)
    if isinstance(vec, np.ndarray) and vec.ndim == 1:
        vec = vec.reshape(1, -1)
    index.add_vectors([mem.id], vec.astype(np.float32))


async def test_consolidate_store_logs_and_raises_on_delete_error(
    store, index, fake_embed, caplog, monkeypatch
) -> None:
    m1 = Memory.new("cat")
    m2 = Memory.new("cat")
    await _add_with_vector(store, index, m1, fake_embed)
    await _add_with_vector(store, index, m2, fake_embed)

    async def fail_delete(_mid: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "delete_memory", fail_delete)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as exc:
        await consolidate_store(store, index, threshold=0.8)

    msg = str(exc.value)
    assert "delete" in msg
    assert m1.id in msg or m2.id in msg
    assert any(
        "delete" in rec.message and (m1.id in rec.message or m2.id in rec.message)
        for rec in caplog.records
    )


async def test_forget_old_memories_logs_and_raises_on_update_error(
    store, index, fake_embed, caplog, monkeypatch
) -> None:
    mem = Memory.new("cat")
    await _add_with_vector(store, index, mem, fake_embed)

    async def fail_update(*_args, **_kwargs) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(store, "update_memory", fail_update)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError) as exc:
        await forget_old_memories(store, index, ttl=0)

    msg = str(exc.value)
    assert "update" in msg
    assert mem.id in msg
    assert any("update" in rec.message and mem.id in rec.message for rec in caplog.records)
