from memory_system import rag_router as rr


def test_reranker_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(rr, "CrossEncoder", None)
    reranker = rr.CrossEncoderReranker("dummy-model")
    docs = [
        rr.MemoryDoc(id="1", text="alpha", channel=rr.Channel.GLOBAL),
        rr.MemoryDoc(id="2", text="beta", channel=rr.Channel.GLOBAL),
    ]
    scores1 = reranker.score("query", docs)
    scores2 = reranker.score("query", docs)
    assert scores1 == [0.0, 0.0]
    assert scores1 == scores2
