"""Tests for maintenance-related Prometheus metrics."""

from __future__ import annotations

import pytest

from memory_system.core.maintenance import consolidate_store, forget_old_memories
from memory_system.core.store import Memory
from memory_system.utils.metrics import (
    LAT_CONSOLIDATION,
    LAT_FORGET,
    MEM_CREATED_TOTAL,
    MEM_DELETED_TOTAL,
)

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)

pytestmark = pytest.mark.asyncio


async def _add(store, index, fake_embed, text: str, *, importance: float = 0.0) -> Memory:
    mem = Memory.new(text, importance=importance)
    await store.add(mem)
    vec = fake_embed(text)
    if isinstance(vec, np.ndarray) and vec.ndim == 1:
        vec = vec.reshape(1, -1)
    index.add_vectors([mem.id], vec.astype(np.float32))
    return mem


async def test_consolidation_metrics(store, index, fake_embed):
    await _add(store, index, fake_embed, "a cat")
    await _add(store, index, fake_embed, "the cat")

    created_before = MEM_CREATED_TOTAL.labels("text")._value.get()
    deleted_before = MEM_DELETED_TOTAL.labels("text")._value.get()
    lat_before = LAT_CONSOLIDATION._count.get()

    created = await consolidate_store(store, index, threshold=0.8)

    assert len(created) == 1
    assert MEM_CREATED_TOTAL.labels("text")._value.get() == created_before + 1
    assert MEM_DELETED_TOTAL.labels("text")._value.get() == deleted_before + 2
    assert LAT_CONSOLIDATION._count.get() == lat_before + 1


async def test_forget_metrics(store, index, fake_embed):
    mems = [await _add(store, index, fake_embed, f"m{i}") for i in range(3)]

    deleted_before = MEM_DELETED_TOTAL.labels("text")._value.get()
    lat_before = LAT_FORGET._count.get()

    deleted = await forget_old_memories(
        store,
        index,
        min_total=0,
        retain_fraction=0.0,
        max_fetch=100,
        chunk_size=10,
    )

    assert deleted == len(mems)
    assert MEM_DELETED_TOTAL.labels("text")._value.get() == deleted_before + len(mems)
    assert LAT_FORGET._count.get() == lat_before + 1
