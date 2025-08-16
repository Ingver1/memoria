import random

import pytest

from scripts.bench_search import DummyStore, bench, fake_embed, make_memories, search_once


@pytest.mark.asyncio
@pytest.mark.perf
async def test_hybrid_not_too_slow():
    random.seed(42)
    memories = make_memories(200)
    store = DummyStore(memories)

    import embedder

    embedder.embed = fake_embed  # type: ignore[attr-defined]

    vec_stats = await bench(lambda: search_once(store, "vector", False, False), reps=5, warmup=2)
    hyb_stats = await bench(lambda: search_once(store, "hybrid", False, False), reps=5, warmup=2)
    assert hyb_stats["median_ms"] <= vec_stats["median_ms"] * 3
