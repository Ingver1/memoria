import time

import pytest

pytest.importorskip("pytest_benchmark")
np = pytest.importorskip("numpy")
from pytest_benchmark.fixture import BenchmarkFixture

from memory_system.rag_router import (
    Channel,
    CompositeScorer,
    CompositeWeights,
    MemoryDoc,
    SimpleRouter,
    mmr_select,
)

pytestmark = [pytest.mark.bench]


def test_router_decide_threshold(benchmark: BenchmarkFixture) -> None:
    router = SimpleRouter()
    result = benchmark(lambda: router.decide("what is python"))
    assert result.stats.stats["mean"] < 0.01


def test_composite_scorer_threshold(benchmark: BenchmarkFixture) -> None:
    scorer = CompositeScorer(CompositeWeights())
    doc = MemoryDoc(
        id="a",
        text="t",
        channel=Channel.GLOBAL,
        last_access_ts=time.time(),
        access_count=0,
        globalness=1.0,
    )
    result = benchmark(lambda: scorer.score(docs=[doc], sim=[0.5], is_global_query=True))
    assert result.stats.stats["mean"] < 0.01


def test_mmr_select_threshold(benchmark: BenchmarkFixture) -> None:
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.eye(3, dtype=np.float32)
    result = benchmark(lambda: mmr_select(base, docs, k=2))
    assert result.stats.stats["mean"] < 0.01
