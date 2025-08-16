"""
Validates real operations of ``EnhancedMemoryStore``:
* adding memories
* retrieving statistics
* semantic search
* reactions to edge cases
"""

import asyncio
import time
from collections.abc import AsyncGenerator

import pytest

import pytest_asyncio

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.store import Memory
from memory_system.settings import RankingConfig, UnifiedSettings
from memory_system.unified_memory import _get_ranking_weights


@pytest_asyncio.fixture(scope="function")
async def store() -> AsyncGenerator[EnhancedMemoryStore, None]:
    """Provide an EnhancedMemoryStore instance for testing."""
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as s:
        yield s


def _rand_embedding(dim: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    arr = rng.random(dim)
    if isinstance(arr, np.ndarray):
        return arr.astype("float32").tolist()
    # fallback: if arr is a float, just wrap in a list
    return [float(arr)]


@pytest.mark.asyncio
async def test_add_and_search(store: EnhancedMemoryStore) -> None:
    """Test adding a memory and searching for it."""
    embedding = _rand_embedding(store.settings.model.vector_dim)
    ts = time.time()

    # 1) add a memory
    await store.add_memory(
        text="hello world",
        role="user",
        tags=["demo"],
        importance=0.2,
        embedding=embedding,
        created_at=ts,
        updated_at=ts,
        valence=0.0,
        emotional_intensity=0.0,
    )

    stats = await store.get_stats()
    assert stats["total_memories"] == 1
    assert stats["index_size"] == 1

    # 2) search using the same embedding
    res = await store.semantic_search(embedding=embedding, k=1)
    assert len(res) == 1
    assert res[0].text == "hello world"

    res_with_dist = await store.semantic_search(embedding=embedding, k=1, return_distance=True)
    assert len(res_with_dist) == 1
    assert res_with_dist[0][0].text == "hello world"
    assert isinstance(res_with_dist[0][1], float)


@pytest.mark.asyncio
async def test_search_empty_store(store: EnhancedMemoryStore) -> None:
    """Test searching when store is empty."""
    embedding = _rand_embedding(store.settings.model.vector_dim)
    res = await store.semantic_search(embedding=embedding, k=1)
    assert res == []


@pytest.mark.asyncio
async def test_invalid_embedding_length(store: EnhancedMemoryStore) -> None:
    """Test that invalid embedding lengths are rejected."""
    # embedding shorter than required 384 elements
    bad_embedding = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        await store.semantic_search(embedding=bad_embedding, k=1)


@pytest.mark.asyncio
async def test_semantic_search_filters_by_level_and_is_faster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    await store.start()

    dim = store.settings.model.vector_dim
    emb0 = np.random.rand(dim).astype("float32").tolist()
    emb1 = np.random.rand(dim).astype("float32").tolist()

    # memory at level 0
    mem0 = Memory.new("lvl0", level=0)
    await store._store.add(mem0)
    store._index.add_vectors("text", [mem0.id], np.asarray([emb0], dtype=np.float32))

    # memory at level 1
    mem1 = Memory.new("lvl1", level=1)
    await store._store.add(mem1)
    store._index.add_vectors("text", [mem1.id], np.asarray([emb1], dtype=np.float32))

    # slow down individual get calls to highlight performance difference
    orig_get = store._store.get

    async def slow_get(mid: str) -> Memory | None:  # type: ignore[override]
        await asyncio.sleep(0.01)
        return await orig_get(mid)

    monkeypatch.setattr(store._store, "get", slow_get)

    async def naive_search(vec: list[float], level: int) -> list[Memory]:
        vec_np = np.asarray(vec, dtype=np.float32)
        ids, _ = store._index.search("text", vec_np, k=5)
        weights = _get_ranking_weights()
        md = {"modality": "text"}
        scored: list[tuple[Memory, float]] = []
        for mid in ids:
            mem = await store._store.get(mid)
            if not mem:
                continue
            if level is not None and mem.level != level:
                continue
            if any(mem.metadata.get(k) != v for k, v in md.items()):
                continue
            score = (
                mem.importance * weights.importance
                + mem.emotional_intensity * weights.emotional_intensity
                + (weights.valence_pos if mem.valence >= 0 else weights.valence_neg) * mem.valence
            )
            scored.append((mem, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:1]]

    start = time.perf_counter_ns()
    old_res = await naive_search(emb0, level=0)
    old_time = (time.perf_counter_ns() - start) / 1e9

    start = time.perf_counter_ns()
    new_res = await store.semantic_search(vector=emb0, k=1, level=0)
    new_time = (time.perf_counter_ns() - start) / 1e9

    assert [m.id for m in new_res] == [m.id for m in old_res]
    assert new_time < old_time

    await store.close()


# Optional extended check
@pytest.mark.asyncio
async def test_duplicate_embeddings_rejected(store: EnhancedMemoryStore) -> None:
    dim = store.settings.model.vector_dim
    v1 = _rand_embedding(dim, seed=1)
    now = time.time()

    # add the first memory
    await store.add_memory(
        text="dup",
        importance=0.1,
        embedding=v1,
        created_at=now,
        updated_at=now,
        role=None,
        tags=None,
        valence=0.0,
        emotional_intensity=0.0,
    )
    # FAISS will not allow the same ID twice,
    # we check duplicate detection at the index level
    with pytest.raises(ValueError):
        store._index.add_vectors([store._index._id_map[1]], np.asarray([v1], dtype=np.float32))


@pytest.mark.asyncio
async def test_semantic_search_respects_min_score() -> None:
    settings = UnifiedSettings.for_testing()
    settings = settings.model_copy(update={"ranking": RankingConfig(min_score=0.5)})
    store = EnhancedMemoryStore(settings)
    await store.start()
    try:
        dim = store.settings.model.vector_dim
        embedding = _rand_embedding(dim)
        now = time.time()

        await store.add_memory(
            text="high",
            importance=0.8,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            valence=0.0,
            emotional_intensity=0.0,
        )
        await store.add_memory(
            text="low",
            importance=0.1,
            embedding=embedding,
            created_at=now,
            updated_at=now,
            valence=0.0,
            emotional_intensity=0.0,
        )

        res = await store.semantic_search(vector=embedding, k=5)
        texts = [m.text for m in res]
        assert "high" in texts
        assert "low" not in texts
    finally:
        await store.close()
