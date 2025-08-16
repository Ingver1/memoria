import datetime as dt
import math

import pytest

from memory_system.unified_memory import Memory, _get_cache, search
from tests.synthetic_dataset import SYNTHETIC_DATASET as DATA


class _FakeVec(list):
    def tolist(self):  # pragma: no cover - trivial
        return list(self)


FAKE_VEC = _FakeVec([0.1, 0.2, 0.3])


class SyntheticStore:
    def __init__(self):
        now = dt.datetime.now(dt.UTC)
        texts = [DATA["correct"], DATA["paraphrase"], *DATA["distractors"]]
        self.memories = [Memory(str(i), t, now) for i, t in enumerate(texts)]
        for m in self.memories:
            m._embedding = FAKE_VEC
        self._map = {m.text: m for m in self.memories}

    async def add_memory(self, memory):  # pragma: no cover - not used
        self.memories.append(memory)

    async def upsert_scores(self, scores):  # pragma: no cover - not used
        return None

    async def search_memory(self, *, query, k=5, metadata_filter=None, level=None, context=None):
        tokens = set(query.lower().split())
        scored = []
        for m in self.memories:
            score = len(tokens & set(m.text.lower().split()))
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    async def semantic_search(
        self,
        *,
        embedding,
        k=5,
        return_distance=False,
        mmr_lambda=None,
        metadata_filter=None,
        level=None,
        context=None,
        modality="text",
    ):
        order = [
            self._map[DATA["paraphrase"]],
            *[self._map[t] for t in DATA["distractors"]],
            self._map[DATA["correct"]],
        ]
        res = [(m, float(i + 1)) for i, m in enumerate(order)]
        sliced = res[:k]
        if return_distance:
            return sliced
        return [m for m, _ in sliced]


def ndcg(results):
    rels = []
    for m in results[:10]:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mmr_lambda", [None, 0.5])
@pytest.mark.parametrize("use_cross", [False, True])
async def test_hybrid_outperforms_vector(mmr_lambda, use_cross, monkeypatch):
    monkeypatch.setattr("embedder.embed", lambda text: FAKE_VEC)
    cache = _get_cache()
    cache.clear()
    store = SyntheticStore()
    reranker = None
    if use_cross:
        from memory_system.settings import UnifiedSettings

        settings = UnifiedSettings.for_testing()
        object.__setattr__(settings.ranking, "use_cross_encoder", True)
        monkeypatch.setattr("memory_system.settings.get_settings", lambda env=None: settings)

        def fake_order(memories, query, **_):
            return sorted(memories, key=lambda m: 0 if m.text == DATA["correct"] else 1)

        monkeypatch.setattr("memory_system.core.cross_reranker.order_by_cross_encoder", fake_order)
        reranker = "cross"

    vec = await search(
        DATA["query"],
        retriever="vector",
        k=10,
        store=store,
        mmr_lambda=mmr_lambda,
        reranker=reranker,
    )
    hyb = await search(
        DATA["query"],
        retriever="hybrid",
        k=10,
        store=store,
        mmr_lambda=mmr_lambda,
        reranker=reranker,
    )
    assert ndcg(hyb) > ndcg(vec)
