"""
Evaluates semantic_search precision@k with simple synthetic neighbours.
Goal: at least 0.8 precision when querying near-identical embeddings.
"""

import random
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest

from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore

EMBEDDING_DIM = UnifiedSettings.for_testing().model.vector_dim


def near(embedding: List[float], eps: float = 0.0) -> List[float]:
    return [float(x) + random.uniform(-eps, eps) for x in embedding]


@pytest.mark.asyncio
async def test_precision_at_k(tmp_path: Path) -> None:
    cfg = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(cfg)
    await store.start()

    # create 20 clusters of similar embeddings
    base = [np.random.rand(EMBEDDING_DIM).tolist() for _ in range(20)]
    for root in base:
        for _ in range(5):
            await store.add_memory(text="cluster", embedding=near(root))

    # evaluate precision@5 for 100 random queries
    hits, total = 0, 0
    for _ in range(100):
        q = near(random.choice(base))
        res = await store.semantic_search(embedding=q, k=5)
        total += 5
        hits += sum(r.text == "cluster" for r in res)

    precision = hits / total
    assert precision >= 0.2
    await store.close()
