import datetime as _dt
import importlib
import sys
import types

import pytest

try:  # pragma: no cover - optional dependency
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - numpy optional
    from tests._stubs import numpy as _np

USING_NUMPY_STUB = bool(getattr(_np, "__stub__", False))

from memory_system.settings import UnifiedSettings
from memory_system.unified_memory import Memory, _get_cache, add, search


class DummyStore:
    def __init__(self) -> None:
        self.memories: list[Memory] = []

    async def add_memory(self, memory: Memory) -> None:  # pragma: no cover - trivial
        self.memories.append(memory)

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter=None,
        level=None,
        context=None,
    ) -> list[Memory]:
        return self.memories[:k]

    async def upsert_scores(self, scores) -> None:  # pragma: no cover - no-op
        return None


@pytest.mark.skipif(_np is None or USING_NUMPY_STUB, reason="numpy not installed")
@pytest.mark.asyncio
async def test_cross_reranker_graceful_fallback(monkeypatch) -> None:
    cache = _get_cache()
    cache.clear()
    store = DummyStore()
    await add("one", store=store)
    await add("two", store=store)
    settings = UnifiedSettings.for_testing()
    object.__setattr__(settings.ranking, "use_cross_encoder", True)
    monkeypatch.setattr("memory_system.settings.get_settings", lambda env=None: settings)
    res = await search("anything", k=2, store=store, reranker="cross")
    assert [m.text for m in res] == ["one", "two"]


def _make_memories() -> list[Memory]:
    now = _dt.datetime.now(_dt.UTC)
    return [
        Memory(memory_id="1", text="one", created_at=now),
        Memory(memory_id="2", text="two", created_at=now),
    ]


def test_cross_reranker_missing_sentence_transformers(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_CROSS_ENCODER", "1")
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    import memory_system.core.cross_reranker as cr

    importlib.reload(cr)
    memories = _make_memories()
    assert cr.order_by_cross_encoder(memories, "query") == memories


def test_cross_reranker_missing_torch(monkeypatch) -> None:
    class DummyCrossEncoder:
        def __init__(self, *_, **__):
            raise ImportError("torch not available")

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=DummyCrossEncoder)
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("ENABLE_CROSS_ENCODER", "1")
    import memory_system.core.cross_reranker as cr

    importlib.reload(cr)
    memories = _make_memories()
    assert cr.order_by_cross_encoder(memories, "query") == memories


def test_cross_reranker_missing_cohere(monkeypatch) -> None:
    monkeypatch.setenv("CROSS_ENCODER_PROVIDER", "cohere")
    monkeypatch.setenv("COHERE_API_KEY", "dummy")
    monkeypatch.delitem(sys.modules, "cohere", raising=False)
    import memory_system.core.cross_reranker as cr

    importlib.reload(cr)
    memories = _make_memories()
    assert cr.order_by_cross_encoder(memories, "query") == memories
