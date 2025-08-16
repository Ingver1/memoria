from unittest.mock import AsyncMock, patch

import pytest

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings


@pytest.mark.asyncio
async def test_add_memory_invalid_embedding_dimension() -> None:
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        bad_embedding = [0.0] * (settings.model.vector_dim - 1)
        with pytest.raises(ValueError):
            await store.add_memory(
                text="bad",
                role=None,
                tags=None,
                importance=0.0,
                valence=0.0,
                emotional_intensity=0.0,
                embedding=bad_embedding,
                created_at=0.0,
                updated_at=0.0,
            )


@pytest.mark.asyncio
async def test_add_memories_batch_invalid_embedding_dimension() -> None:
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        bad_embedding = [0.0] * (settings.model.vector_dim - 1)
        items = [
            {
                "text": "bad",
                "role": None,
                "tags": None,
                "importance": 0.0,
                "valence": 0.0,
                "emotional_intensity": 0.0,
                "embedding": bad_embedding,
                "created_at": 0.0,
                "updated_at": 0.0,
            }
        ]
        with pytest.raises(ValueError):
            await store.add_memories_batch(items)


@pytest.mark.asyncio
async def test_add_memory_non_numeric_embedding() -> None:
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        bad_embedding = ["x"] * settings.model.vector_dim
        with pytest.raises(ValueError):
            await store.add_memory(
                text="bad",
                role=None,
                tags=None,
                importance=0.0,
                valence=0.0,
                emotional_intensity=0.0,
                embedding=bad_embedding,
                created_at=0.0,
                updated_at=0.0,
            )


@pytest.mark.asyncio
async def test_add_memory_database_failure() -> None:
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        vec = list(np.random.rand(settings.model.vector_dim).astype(np.float32))
        with (
            patch.object(
                store._store,
                "add",
                new=AsyncMock(side_effect=OSError("db down")),
            ),
            pytest.raises(OSError),
        ):
            await store.add_memory(
                text="fail",
                role=None,
                tags=None,
                importance=0.0,
                valence=0.0,
                emotional_intensity=0.0,
                embedding=vec,
                created_at=0.0,
                updated_at=0.0,
            )
