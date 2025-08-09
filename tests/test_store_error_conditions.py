from unittest.mock import AsyncMock, patch

import pytest

np = pytest.importorskip("numpy")

from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.index import ANNIndexError


@pytest.mark.asyncio
async def test_add_memory_invalid_embedding_dimension() -> None:
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    await store.start()
    try:
        bad_embedding = [0.0] * (settings.model.vector_dim - 1)
        with pytest.raises(ANNIndexError):
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
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_add_memory_non_numeric_embedding() -> None:
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    await store.start()
    try:
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
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_add_memory_database_failure() -> None:
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    await store.start()
    try:
        vec = list(np.random.rand(settings.model.vector_dim).astype(np.float32))
        with patch.object(
            store._store,
            "add",
            new=AsyncMock(side_effect=OSError("db down")),
        ):
            with pytest.raises(OSError):
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
    finally:
        await store.close()
