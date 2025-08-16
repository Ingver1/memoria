import datetime as dt
import json
import math
from pathlib import Path

import pytest

from memory_system.unified_memory import Memory, search

DATA_DIR = Path(__file__).parent / "data"


class MiniStore:
    def __init__(self, docs):
        now = dt.datetime.now(dt.UTC)
        self.memories = [Memory(doc_id, text, now) for doc_id, text in docs.items()]

    async def add_memory(self, memory):  # pragma: no cover - unused
        self.memories.append(memory)

    async def upsert_scores(self, scores):  # pragma: no cover - unused
        return None

    async def search_memory(self, *, query, k=5, metadata_filter=None, level=None, context=None):
        tokens = set(query.lower().split())
        scored = []
        for m in self.memories:
            score = len(tokens & set(m.text.lower().split()))
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]


def load_dataset(dataset: str, lang: str):
    base = DATA_DIR / dataset / lang
    docs = {}
    with open(base / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs[obj["_id"]] = obj["text"]
    queries = {}
    with open(base / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]
    qrels = {}
    with open(base / "qrels.tsv", encoding="utf-8") as f:
        for line in f:
            qid, _zero, doc_id, rel = line.strip().split("\t")
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return docs, queries, qrels


def ndcg_at_10(rels):
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:10]))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:10]))
    return dcg / idcg if idcg else 0.0


DATASETS = [
    ("beir", "en"),
    ("beir", "es"),
    ("beir", "ja"),
    ("miracl", "en"),
    ("miracl", "es"),
    ("miracl", "ja"),
]

THRESHOLDS = {
    "beir": {"en": 0.9, "es": 0.9, "ja": 0.9},
    "miracl": {"en": 0.8, "es": 0.8, "ja": 0.8},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset,lang", DATASETS)
async def test_multilingual_ir(dataset, lang):
    docs, queries, qrels = load_dataset(dataset, lang)
    store = MiniStore(docs)
    scores = []
    for qid, query in queries.items():
        res = await search(query, retriever="sparse", k=10, store=store)
        rels = [qrels.get(qid, {}).get(m.memory_id, 0) for m in res]
        scores.append(ndcg_at_10(rels))
    avg = sum(scores) / len(scores) if scores else 0.0
    assert avg >= THRESHOLDS[dataset][lang]
