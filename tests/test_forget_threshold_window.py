import datetime as dt

import pytest

from memory_system.core.maintenance import forget_old_memories
from memory_system.core.store import Memory

np = pytest.importorskip("numpy")
faiss = pytest.importorskip("faiss")
if faiss.__name__.startswith("tests._stubs"):
    pytest.skip("faiss not installed", allow_module_level=True)


@pytest.mark.asyncio
async def test_forget_requires_long_low_score(store, index, fake_embed):
    now = dt.datetime.now(dt.UTC)
    old_ts = (now - dt.timedelta(days=20)).isoformat()
    recent_ts = (now - dt.timedelta(days=5)).isoformat()

    old_low = Memory.new(
        "old low",
        importance=0.0,
        valence=0.0,
        emotional_intensity=0.0,
        metadata={"high_score_ts": old_ts, "trust_score": 0.0},
    )
    recent_low = Memory.new(
        "recent low",
        importance=0.0,
        valence=0.0,
        emotional_intensity=0.0,
        metadata={"high_score_ts": recent_ts, "trust_score": 0.0},
    )
    with_evidence = Memory.new(
        "evidence",
        importance=0.0,
        valence=0.0,
        emotional_intensity=0.0,
        metadata={"high_score_ts": old_ts, "has_evidence": True, "trust_score": 0.0},
    )
    high = Memory.new(
        "high",
        importance=1.0,
        valence=1.0,
        emotional_intensity=1.0,
        metadata={"trust_score": 0.0},
    )

    for mem in (old_low, recent_low, with_evidence, high):
        await store.add(mem)
    vecs = fake_embed([m.text for m in (old_low, recent_low, with_evidence, high)])
    if isinstance(vecs, np.ndarray) and vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    index.add_vectors(
        [m.id for m in (old_low, recent_low, with_evidence, high)],
        vecs.astype(np.float32),
    )

    deleted = await forget_old_memories(
        store,
        index,
        min_total=0,
        retain_fraction=1.0,
    )
    assert deleted == 1
    remaining = {m.id for m in await store.search(limit=10)}
    assert remaining == {recent_low.id, with_evidence.id, high.id}
    stored = next(m for m in await store.search(limit=10) if m.id == high.id)
    assert "high_score_ts" in stored.metadata
