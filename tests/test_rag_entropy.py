import math
from types import SimpleNamespace

import pytest

from memory_system import rag_router
from memory_system.rag_router import (
    Channel,
    PipelineConfig,
    RAGRouterPipeline,
    RouterDecision,
    calc_entropy,
)


def _make_pipeline(monkeypatch: pytest.MonkeyPatch) -> RAGRouterPipeline:
    class DummyClient:
        def __init__(self, *a, **k):
            pass

    class DummyTimeout:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(
        rag_router,
        "httpx",
        type("x", (), {"Client": DummyClient, "Timeout": DummyTimeout}),
    )

    class DummyStore:
        def __init__(self, dim: int, **_: object) -> None:
            self.docs = []

        def add(self, docs, vectors):
            self.docs.extend(docs)

        def mark_access(self, ids, ts=None):
            pass

        def search(self, qvec, k):
            return [], []

    monkeypatch.setattr(rag_router, "LocalPersonalStore", DummyStore)
    cfg = PipelineConfig(base_url="http://localhost")
    return RAGRouterPipeline(cfg)


def test_calc_entropy() -> None:
    ent = calc_entropy([0.5, 0.5])
    assert math.isclose(ent, math.log(2), rel_tol=1e-6)


def test_retrieve_skips_on_low_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch)
    called = False

    def dummy_decide(query: str, project_context: bool = False) -> RouterDecision:
        nonlocal called
        called = True
        return RouterDecision(
            use_retrieval=True,
            target_channels=[Channel.GLOBAL],
            is_global_query=True,
            reason="router",
        )

    pipe.router.decide = dummy_decide  # type: ignore[assignment]
    monkeypatch.setattr(pipe, "_lm_token_probs", lambda q: [1.0])
    settings = SimpleNamespace(rag=SimpleNamespace(entropy_threshold=0.5))
    monkeypatch.setattr(rag_router, "get_settings", lambda: settings)

    docs, decision = pipe.retrieve("hello")
    assert docs == []
    assert decision.use_retrieval is False
    assert called is False


def test_retrieve_calls_router_on_high_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch)
    called = False

    def dummy_decide(query: str, project_context: bool = False) -> RouterDecision:
        nonlocal called
        called = True
        return RouterDecision(
            use_retrieval=False,
            target_channels=[],
            is_global_query=True,
            reason="router",
        )

    pipe.router.decide = dummy_decide  # type: ignore[assignment]
    monkeypatch.setattr(pipe, "_lm_token_probs", lambda q: [0.25, 0.25, 0.25, 0.25])
    settings = SimpleNamespace(rag=SimpleNamespace(entropy_threshold=0.5))
    monkeypatch.setattr(rag_router, "get_settings", lambda: settings)

    docs, decision = pipe.retrieve("hello")
    assert docs == []
    assert decision.reason == "router"
    assert called is True
