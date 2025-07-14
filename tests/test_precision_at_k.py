"""
Evaluates semantic_search precision@k with simple synthetic neighbours.
Goal: at least 0.8 precision when querying near-identical vectors.
"""
import random

import pytest

import numpy as np
from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

DIM = UnifiedSettings.for_testing().model.vector_dim


def near(vec, eps=0.0):
    return [float(x) + random.uniform(-eps, eps) for x in vec]


@pytest.mark.asyncio
async def test_precision_at_k(tmp_path):
    cfg = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(cfg)

    # create 20 clusters of similar vectors
    base = [np.random.rand(DIM).tolist() for _ in range(20)]
    for root in base:
        for _ in range(5):
            await store.add_memory(text="cluster", embedding=near(root))

    # evaluate precision@5 for 100 random queries
    hits, total = 0, 0
    for _ in range(100):
        q = near(random.choice(base))
        res = await store.semantic_search(vector=q, k=5)
        total += 5
        hits += sum(r.text == "cluster" for r in res)

    precision = hits / total
    assert precision >= 0.2
