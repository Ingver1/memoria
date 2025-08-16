import datetime as dt

import pytest

from memory_system.unified_memory import Memory, search


class DummyStore:
    def __init__(self, memories):
        self.memories = memories

    async def search_memory(self, *, query, k=5, metadata_filter=None, level=None, context=None):
        return list(self.memories)


@pytest.mark.asyncio
async def test_mmr_deduplicates_paraphrases(monkeypatch):
    vecs = {
        "query": [1.0, 0.0, 0.0],
        "dup1": [0.8, 0.6, 0.0],
        "dup2": [0.8107, 0.5855, 0.0],
        "other": [0.7, -0.7141428, 0.0],
    }

    def fake_embed(text):
        return vecs[text]

    monkeypatch.setattr("embedder.embed", fake_embed)

    now = dt.datetime.now(dt.UTC)
    mems = [
        Memory("1", "dup1", now),
        Memory("2", "dup2", now),
        Memory("3", "other", now),
    ]
    store = DummyStore(mems)

    baseline = await search("query", k=2, store=store, mmr_lambda=None)
    assert [m.text for m in baseline[:2]] == ["dup1", "dup2"]

    deduped = await search("query", k=2, store=store)
    assert [m.text for m in deduped[:2]] == ["dup2", "other"]
