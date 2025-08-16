import pytest

from memory_system import rag_router
from memory_system.rag_router import (
    Channel,
    MemoryDoc,
    PipelineConfig,
    RAGRouterPipeline,
)


def _make_pipeline(monkeypatch: pytest.MonkeyPatch) -> RAGRouterPipeline:
    class DummyClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(rag_router, "httpx", type("x", (), {"Client": DummyClient}))

    class DummyStore:
        def __init__(self, dim: int, **_: object) -> None:
            self.docs: list[MemoryDoc] = []

        def add(self, docs, vectors) -> None:  # type: ignore[no-untyped-def]
            self.docs.extend(docs)

        def mark_access(self, ids, ts=None) -> None:  # type: ignore[no-untyped-def]
            pass

        def search(self, qvec, k):  # type: ignore[no-untyped-def]
            return list(self.docs), []

    monkeypatch.setattr(rag_router, "LocalPersonalStore", DummyStore)
    cfg = PipelineConfig(base_url="http://localhost")
    return RAGRouterPipeline(cfg)


def test_mark_success_updates_metadata_and_reranks(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch)
    doc_a = MemoryDoc(id="a", text="x", channel=Channel.PERSONAL)
    doc_b = MemoryDoc(id="b", text="x", channel=Channel.PERSONAL)
    pipe.personal_store.docs.extend([doc_a, doc_b])
    pipe._last_results = [doc_a, doc_b]
    pipe.mark_success("b")
    assert doc_b.metadata["success_count"] == 1
    assert doc_b.metadata["trial_count"] == 1
    assert [d.id for d in pipe._last_results] == ["b", "a"]


def test_mark_failure_increments_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch)
    doc_a = MemoryDoc(id="a", text="x", channel=Channel.PERSONAL)
    pipe.personal_store.docs.append(doc_a)
    pipe._last_results = [doc_a]
    pipe.mark_failure("a")
    assert doc_a.metadata["trial_count"] == 1
    assert doc_a.metadata.get("success_count", 0) == 0
