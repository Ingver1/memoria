import os

import pytest

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore


@pytest.mark.asyncio
async def test_enhanced_store_add_search_list_stats(tmp_path):
    os.environ["DATABASE__DB_PATH"] = str(tmp_path / "mem.db")
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    try:
        vec_dim = settings.model.vector_dim
        emb1 = list(np.random.rand(vec_dim).astype(np.float32))
        emb2 = list(np.random.rand(vec_dim).astype(np.float32))
        now = 0.0

        mem1 = await store.add_memory(
            text="hello",
            role="user",
            tags=["test"],
            importance=0.5,
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
            importance=0.5,
            valence=0.0,
            emotional_intensity=0.0,
            embedding=emb2,
            created_at=now,
            updated_at=now,
        )

        memories = await store.list_memories()
        assert len(memories) == 2

        results = await store.semantic_search(vector=emb1, k=1)
        assert results and results[0].id == mem1.id

        stats = await store.get_stats()
        assert stats["total_memories"] == 2
        assert stats["index_size"] == 2
    finally:
        await store.close()
