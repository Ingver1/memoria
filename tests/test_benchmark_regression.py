"""Benchmark regression tests.

These tests fail CI if performance regresses by more than 10 % compared
with the stored baseline. pytest-benchmark handles the comparison, and
pytest_asyncio is used for async fixtures.
"""
import asyncio

import pytest

import pytest_asyncio

# Skip this test module if pytest-benchmark is unavailable
pytest.importorskip("pytest_benchmark")

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim
VECTOR = np.random.rand(DIM).astype("float32").tolist()


@pytest_asyncio.fixture(scope="session")
async def bench_store():
    s = EnhancedMemoryStore(UnifiedSettings.for_testing())
    for _ in range(1_000):
        await s.add_memory(text="bench", embedding=np.random.rand(DIM).tolist())
    yield s
    await s.close()


@pytest.mark.perf
@pytest.mark.benchmark
def test_semantic_search_speed(mem_benchmark, bench_store):
    async def _run() -> None:
        asyncio.get_event_loop()
        await bench_store.semantic_search(vector=VECTOR, k=5)

    mem_benchmark(lambda: asyncio.run(_run()))
