import asyncio
import threading
import time

import pytest

from memory_system.core import enhanced_store as enhanced_store_module
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings
from memory_system.utils import numpy as numpy_utils


class DummyArray(list):
    def __init__(self, data):
        super().__init__(data)
        self.ndim = 1
        self.shape = (len(data),)


class DummyNP:
    float32 = "float32"

    @staticmethod
    def asarray(x, dtype=None):
        return DummyArray(x)


@pytest.mark.asyncio
async def test_semantic_search_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semantic search serialises only when a non-thread-safe index is in use."""
    monkeypatch.setattr(numpy_utils, "require_numpy", lambda: DummyNP)
    monkeypatch.setattr(enhanced_store_module, "require_numpy", lambda: DummyNP)
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        calls: dict[str, int] = {"active": 0, "max": 0}
        lock = threading.Lock()

        def fake_search(*args, **kwargs):
            with lock:
                calls["active"] += 1
                calls["max"] = max(calls["max"], calls["active"])
            time.sleep(0.05)
            with lock:
                calls["active"] -= 1
            return [], []

        store.vector_store.search = fake_search  # type: ignore[assignment]
        emb = [0.0] * store.settings.model.vector_dim

        async def run_search() -> None:
            await store.semantic_search(embedding=emb, k=1)

        await asyncio.gather(*(run_search() for _ in range(5)))

        if store._search_lock is None:
            assert calls["max"] >= 2
        else:
            assert calls["max"] == 1


@pytest.mark.asyncio
async def test_evaluate_recall_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recall evaluation search concurrency mirrors the index's thread safety."""
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:
        calls: dict[str, int] = {"active": 0, "max": 0}
        lock = threading.Lock()

        def fake_search(*args, **kwargs):
            with lock:
                calls["active"] += 1
                calls["max"] = max(calls["max"], calls["active"])
            time.sleep(0.05)
            with lock:
                calls["active"] -= 1
            return [], []

        store.vector_store.search = fake_search  # type: ignore[assignment]
        zero_vec = [0.0] * store.settings.model.vector_dim
        store._control_queries = [(zero_vec, set())]

        await asyncio.gather(*(store._evaluate_recall() for _ in range(5)))

        if store._search_lock is None:
            assert calls["max"] >= 2
        else:
            assert calls["max"] == 1
