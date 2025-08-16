import importlib.util

import pytest

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("numpy not installed", allow_module_level=True)

from memory_system.core import maintenance as m

pytestmark = pytest.mark.needs_numpy

_spec = importlib.util.find_spec("faiss")
_HAS_FAISS = bool(_spec and "tests/_stubs" not in (_spec.origin or ""))


def test_cluster_memories_auto_switch(monkeypatch):
    calls: list[str] = []

    def fake_greedy(embeddings, threshold):
        calls.append("greedy")
        return [[i] for i in range(len(embeddings))]

    def fake_faiss(embeddings, threshold):
        calls.append("faiss")
        return [[i] for i in range(len(embeddings))]

    monkeypatch.setattr(m, "cluster_memories", fake_greedy)
    monkeypatch.setattr(m, "cluster_memories_faiss", fake_faiss)

    small = [np.zeros(3, dtype=np.float32) for _ in range(5)]
    m.cluster_memories_auto(small, algorithm="auto", large_threshold=10)
    assert calls == ["greedy"]

    calls.clear()
    large = [np.zeros(3, dtype=np.float32) for _ in range(20)]
    m.cluster_memories_auto(large, algorithm="auto", large_threshold=10)
    assert calls == ["faiss"]


@pytest.mark.skipif(not _HAS_FAISS, reason="faiss not installed")
def test_cluster_memories_faiss_basic():
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.9, 0.1], dtype=np.float32)
    e3 = np.array([0.0, 1.0], dtype=np.float32)
    clusters = m.cluster_memories_faiss([e1, e2, e3], threshold=0.5, n_clusters=2)
    assert sum(len(c) for c in clusters) == 3


def test_cluster_memories_faiss_empty():
    assert m.cluster_memories_faiss([]) == []
