"""Micro-benchmark for semantic search performance.

Measures queries-per-second for different ``ef_search`` values to help
pick sensible defaults for the FAISS index. Async fixtures are provided
via ``pytest_asyncio``.
"""
import asyncio
from typing import Any, AsyncGenerator

import pytest

import pytest_asyncio
from memory_system.utils.loop import get_or_create_loop
from pytest_benchmark.fixture import BenchmarkFixture

# Skip benchmarks when pytest-benchmark is not installed
pytest.importorskip("pytest_benchmark")

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim
RANDOM_VECTOR = np.random.rand(DIM).astype("float32").tolist()


@pytest_asyncio.fixture(scope="session")
async def populated_store() -> AsyncGenerator[EnhancedMemoryStore, None]:
    """Fill the index with 2 000 random vectors to make the test realistic."""
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    for _ in range(2_000):
        await store.add_memory(
            text="dummy",
            embedding=np.random.rand(DIM).astype("float32").tolist(),
        )
    yield store
    await store.close()


@pytest.mark.perf
@pytest.mark.benchmark
@pytest.mark.parametrize("ef", [10, 50, 100, 200])
def test_benchmark_semantic_search(benchmark: BenchmarkFixture, populated_store: EnhancedMemoryStore, ef: int) -> None:
    """Benchmark throughput using pytest-benchmark."""

    loop = get_or_create_loop()
    benchmark(
        lambda: loop.run_until_complete(
            populated_store.semantic_search(
                vector=RANDOM_VECTOR,
                k=5,
                ef_search=ef,
            )
        )
    )
    
