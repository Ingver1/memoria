import pytest

np = pytest.importorskip("numpy")

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


@pytest.mark.asyncio
async def test_attention_reranks_by_keyword() -> None:
    cache = _get_cache()
    cache.clear()
    store = DummyStore()

    await add("unrelated text", store=store)
    await add("attention keyword here", store=store)

    plain = await search("anything", k=2, store=store)
    assert [m.text for m in plain] == ["unrelated text", "attention keyword here"]

    reranked = await search(
        "anything", k=2, store=store, reranker="keyword", attention_query="keyword"
    )
    assert [m.text for m in reranked] == ["attention keyword here", "unrelated text"]
    assert reranked[0]._tokens is not None
