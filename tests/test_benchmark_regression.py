"""Benchmark regression tests.

These tests fail CI if performance regresses by more than 10 % compared
with the stored baseline. pytest-benchmark handles the comparison, and
pytest_asyncio is used for async fixtures.
"""

import asyncio
from typing import Any, AsyncGenerator, Callable

import numpy as np
import pytest
import pytest_asyncio

from memory_system.utils.loop import get_or_create_loop

# Skip this test module if pytest-benchmark is unavailable
pytest.importorskip("pytest_benchmark")

from memory_system.config.settings import UnifiedSettings
from memory_system.core.embedding import EmbeddingError, EmbeddingService
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.utils.metrics import EMBEDDING_QUEUE_LENGTH

DIM = UnifiedSettings.for_testing().model.vector_dim
EMBEDDING = np.random.rand(DIM).astype("float32").tolist()


@pytest_asyncio.fixture(scope="session")
async def bench_store() -> AsyncGenerator[EnhancedMemoryStore, None]:
    s = EnhancedMemoryStore(UnifiedSettings.for_testing())
    await s.start()
    for _ in range(1_000):
        await s.add_memory(text="bench", embedding=np.random.rand(DIM).tolist())
    yield s
    await s.close()


@pytest.mark.perf
@pytest.mark.benchmark
def test_semantic_search_speed(benchmark: Any, bench_store: EnhancedMemoryStore) -> None:
    def _run() -> None:
        get_or_create_loop()
        asyncio.run(bench_store.semantic_search(embedding=EMBEDDING, k=5))

    benchmark(_run)


@pytest.mark.asyncio
async def test_embedding_queue_limit() -> None:
    cfg = UnifiedSettings.for_testing()
    cfg.performance.queue_max_size = 1
    svc = EmbeddingService(cfg.model.model_name, cfg)
    try:
        t1 = asyncio.create_task(svc.embed_text("a"))
        await asyncio.sleep(0.01)
        assert EMBEDDING_QUEUE_LENGTH._value.get() == 1
        with pytest.raises(EmbeddingError):
            await svc.encode("b")
        await t1
    finally:
        svc.shutdown()
