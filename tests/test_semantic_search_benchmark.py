"""
Micro-benchmark for semantic search performance.

Measures queries-per-second for different ``ef_search`` values to help
pick sensible defaults for the FAISS index. Async fixtures are provided
via ``pytest_asyncio``.
"""

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import pytest

import pytest_asyncio

# Skip benchmarks when pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")
if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
else:  # pragma: no cover - used only when plugin missing
    BenchmarkFixture = Any  # type: ignore

from memory_system.utils.loop import get_loop

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings

DIM = UnifiedSettings.for_testing().model.vector_dim
RANDOM_EMBEDDING = np.random.rand(DIM).astype("float32").tolist()


@pytest_asyncio.fixture(scope="session")
async def populated_store() -> AsyncGenerator[EnhancedMemoryStore, None]:
    """Fill the index with 2 000 random embeddings to make the test realistic."""
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        for _ in range(2_000):
            await store.add_memory(
                text="dummy",
                embedding=np.random.rand(DIM).astype("float32").tolist(),
            )
        yield store


@pytest.mark.perf
@pytest.mark.bench
@pytest.mark.parametrize("ef", [10, 50, 100, 200])
def test_benchmark_semantic_search(
    benchmark: BenchmarkFixture, populated_store: EnhancedMemoryStore, ef: int
) -> None:
    """Benchmark throughput using pytest-benchmark."""
    loop = get_loop()
    benchmark(
        lambda: loop.run_until_complete(
            populated_store.semantic_search(
                embedding=RANDOM_EMBEDDING,
                k=5,
                ef_search=ef,
            )
        )
    )

