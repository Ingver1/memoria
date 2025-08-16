import datetime as dt
import math

import pytest

from memory_system.core import cross_reranker as cr
from memory_system.unified_memory import Memory
from tests.synthetic_dataset import SYNTHETIC_DATASET as DATA


def _make_memories():
    now = dt.datetime.now(dt.UTC)
    return [
        Memory("1", DATA["paraphrase"], now),
        Memory("2", DATA["distractors"][0], now),
        Memory("3", DATA["correct"], now),
    ]


def ndcg(results):
    rels = []
    for m in results:
        if m.text == DATA["correct"]:
            rels.append(3)
        elif m.text == DATA["paraphrase"]:
            rels.append(1)
        else:
            rels.append(0)
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal_rels = sorted([3, 1] + [0] * (len(rels) - 2), reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg else 0.0


def position(results):
    for i, m in enumerate(results, 1):
        if m.text == DATA["correct"]:
            return i
    return 0


@pytest.mark.asyncio
async def test_cross_reranker_improves_quality(monkeypatch):
    memories = _make_memories()
    base_ndcg = ndcg(memories)
    base_pos = position(memories)

    class StubModel:
        def predict(self, pairs):  # pragma: no cover - trivial
            return [0.5, 0.1, 0.9]

    monkeypatch.setattr(cr, "_load_model", lambda: StubModel())
    reranked = cr.order_by_cross_encoder(memories, DATA["query"])
    rerank_ndcg = ndcg(reranked)
    rerank_pos = position(reranked)

    assert rerank_ndcg > base_ndcg
    assert rerank_pos < base_pos
