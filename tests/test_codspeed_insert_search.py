"""CodSpeed benchmark examples for insert and search."""

import asyncio

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pytest_codspeed")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings

DIM = UnifiedSettings.for_testing().model.vector_dim
EMBEDDING = np.random.rand(DIM).astype("float32").tolist()


@pytest.mark.perf
@pytest.mark.bench
def test_insert_benchmark(codspeed) -> None:
    """Benchmark memory insertion using CodSpeed."""

    async def _insert() -> None:
        async with EnhancedMemoryStore(UnifiedSettings.for_testing()) as store:
            await store.add_memory(text="bench", embedding=EMBEDDING)

    from memory_system.utils.loop import get_loop

    codspeed.benchmark(lambda: get_loop().run_until_complete(_insert()))


@pytest.mark.perf
@pytest.mark.bench
def test_search_benchmark(codspeed) -> None:
    """Benchmark semantic search using CodSpeed."""

    async def _search() -> None:
        async with EnhancedMemoryStore(UnifiedSettings.for_testing()) as store:
            await store.add_memory(text="bench", embedding=EMBEDDING)
            await store.semantic_search(embedding=EMBEDDING, k=1)

    from memory_system.utils.loop import get_loop

    codspeed.benchmark(lambda: get_loop().run_until_complete(_search()))
