import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings


@pytest.mark.asyncio
async def test_enhanced_store_add_search_list_stats(tmp_path: Path) -> None:
    """Test adding memories, searching, listing and retrieving stats from enhanced store."""
    os.environ["DATABASE__DB_PATH"] = str(tmp_path / "mem.db")
    os.environ["DATABASE__VEC_PATH"] = str(tmp_path / "mem.vectors")
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        vec_dim = settings.model.vector_dim
        emb1 = list(np.random.rand(vec_dim).astype(np.float32))
        emb2 = list(np.random.rand(vec_dim).astype(np.float32))
        now = 0.0

        mem1 = await store.add_memory(
            text="hello",
            role="user",
            tags=["test"],
            importance=0.7,
            valence=0.0,
            emotional_intensity=0.0,
            embedding=emb1,
            created_at=now,
            updated_at=now,
        )
        await store.add_memory(
            text="world",
            role="user",
            tags=[],
            importance=0.3,
            valence=0.0,
            emotional_intensity=0.0,
            embedding=emb2,
            created_at=now,
            updated_at=now,
        )

        memories = await store.list_memories()
        assert len(memories) == 2

        results = await store.semantic_search(embedding=emb1, k=1)
        assert results and results[0].id == mem1.id

        results_with_dist = await store.semantic_search(embedding=emb1, k=1, return_distance=True)
        assert any(m.id == mem1.id for m, _ in results_with_dist)
        returned_dist = next(d for m, d in results_with_dist if m.id == mem1.id)
        assert isinstance(returned_dist, float)
        assert np.isclose(returned_dist, 0.0)

        stats = await store.get_stats()
        assert stats["total_memories"] == 2
        assert stats["index_size"] == 2

    assert settings.database.vec_path.exists()

    async with EnhancedMemoryStore(settings) as store:
        results_after_reload = await store.semantic_search(embedding=emb1, k=1)
        assert results_after_reload and results_after_reload[0].id == mem1.id
