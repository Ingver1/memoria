import pytest

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)

from memory_system.core.maintenance import _decay_score, forget_old_memories
from memory_system.core.store import Memory


@pytest.mark.asyncio
async def test_forget_by_low_trust(store, index, fake_embed):
    low = Memory.new("low", metadata={"trust_score": 0.1})
    high = Memory.new("high", metadata={"trust_score": 0.9})
    await store.add(low)
    await store.add(high)
    vecs = fake_embed([low.text, high.text])
    if isinstance(vecs, np.ndarray) and vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    index.add_vectors([low.id, high.id], vecs.astype(np.float32))
    deleted = await forget_old_memories(
        store,
        index,
        min_total=0,
        retain_fraction=1.0,
        low_trust=0.2,
    )
    assert deleted == 1
    remaining = await store.search(limit=10)
    assert remaining and remaining[0].id == high.id


def test_decay_score_trust_penalty():
    full = _decay_score(
        importance=1.0,
        valence=1.0,
        emotional_intensity=1.0,
        trust=1.0,
        age_days=0.0,
    )
    low = _decay_score(
        importance=1.0,
        valence=1.0,
        emotional_intensity=1.0,
        trust=0.5,
        age_days=0.0,
    )
    assert low < full
