import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)

import memory_system.core.hierarchical_summarizer as hs
from memory_system.core.hierarchical_summarizer import HierarchicalSummarizer
from memory_system.core.store import Memory

pytestmark = pytest.mark.asyncio


async def test_singleton_clusters_are_marked_final(store, index, fake_embed, monkeypatch):
    monkeypatch.setattr(hs, "embed_text", fake_embed, raising=True)
    mem = Memory.new("a lonely cat", valence=0.3, emotional_intensity=0.8)
    await store.add(mem)
    vec = fake_embed(mem.text)
    if isinstance(vec, np.ndarray) and vec.ndim == 1:
        vec = vec.reshape(1, -1)
    index.add_vectors([mem.id], vec.astype(np.float32))

    summarizer = HierarchicalSummarizer(store, index)

    level = 0
    for _ in range(5):
        created = await summarizer.build_level(level)
        if not created:
            break
        level += 1

    assert level == 0
    # memory is marked final and no higher level memories exist
    stored = await store.get(mem.id)
    assert stored and stored.metadata.get("final") is True
    assert stored.valence == pytest.approx(0.3)
    assert stored.emotional_intensity == pytest.approx(0.8)
    assert await store.search(limit=10, level=1) == []


async def test_cluster_summary_carries_emotions(store, index, fake_embed, monkeypatch):
    monkeypatch.setattr(hs, "embed_text", fake_embed, raising=True)
    mem1 = Memory.new(
        "a cat",
        importance=0.2,
        valence=0.4,
        emotional_intensity=0.6,
        metadata={"trust_score": 0.8},
    )
    mem2 = Memory.new(
        "the cat",
        importance=0.4,
        valence=-0.2,
        emotional_intensity=0.2,
        metadata={"trust_score": 0.4},
    )
    for m in (mem1, mem2):
        await store.add(m)
        vec = fake_embed(m.text)
        if isinstance(vec, np.ndarray) and vec.ndim == 1:
            vec = vec.reshape(1, -1)
        index.add_vectors([m.id], vec.astype(np.float32))

    summarizer = HierarchicalSummarizer(store, index)
    created = await summarizer.build_level(0)
    assert len(created) == 1
    summary = created[0]
    assert summary.valence == pytest.approx(np.mean([mem1.valence, mem2.valence]))
    assert summary.emotional_intensity == pytest.approx(
        np.mean([mem1.emotional_intensity, mem2.emotional_intensity])
    )
    assert summary.metadata.get("trust_score") == pytest.approx(np.mean([0.8, 0.4]))
