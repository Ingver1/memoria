import pytest

import memory_system.unified_memory as um
from memory_system.unified_memory import DuplicateMemoryError, Memory


class DummyStore:
    def __init__(self, memories: list[Memory]):
        self.memories = memories

    async def search_memory(self, *, query, k=5, metadata_filter=None, level=None, context=None):
        return list(self.memories)

    async def add_memory(self, memory: Memory) -> None:
        self.memories.append(memory)

    async def upsert_scores(self, scores):
        return None

    async def close(self):
        await um.close()


@pytest.mark.asyncio
async def test_fast_duplicate_rejects_without_cross_encoder(monkeypatch):
    monkeypatch.setattr(um, "_deep_dedup_enabled", lambda: False)
    monkeypatch.setattr(um, "_get_min_score", lambda: 0.0)

    calls = {"n": 0}

    def fake_score(text: str) -> float:
        calls["n"] += 1
        return 1.0

    monkeypatch.setattr(um, "_cross_encoder_score", fake_score)

    store = DummyStore([])
    await um.add("Hello world", store=store)
    with pytest.raises(DuplicateMemoryError):
        await um.add("hello   world!!!", store=store)
    assert calls["n"] == 1
    await store.close()


@pytest.mark.asyncio
async def test_add_rejects_low_score(monkeypatch):
    monkeypatch.setattr(um, "_deep_dedup_enabled", lambda: False)
    monkeypatch.setattr(um, "_cross_encoder_score", lambda text: 0.1)
    monkeypatch.setattr(um, "_get_min_score", lambda: 0.5)

    store = DummyStore([])
    with pytest.raises(ValueError, match="draft score"):
        await um.add("unique text", store=store)
    await store.close()


@pytest.mark.asyncio
async def test_verifier_hook(monkeypatch):
    monkeypatch.setattr(um, "_deep_dedup_enabled", lambda: False)
    monkeypatch.setattr(um, "_cross_encoder_score", lambda text: 1.0)
    monkeypatch.setattr(um, "_get_min_score", lambda: 0.0)

    um._VERIFIERS.clear()
    um.register_verifier("episodic", lambda text: "good" in text)

    store = DummyStore([])
    with pytest.raises(ValueError, match="verification failed"):
        await um.add("bad text", store=store)

    um._VERIFIERS.clear()
    await store.close()
