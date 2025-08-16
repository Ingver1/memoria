import pytest

from memory_system.core.redis_store import RedisVectorStore


class DummyClient:
    pass


@pytest.mark.asyncio
async def test_add_dimension_mismatch_raises() -> None:
    store = object.__new__(RedisVectorStore)
    store._dim = 3
    store._client = DummyClient()  # type: ignore[attr-defined]
    store._index = "idx"
    with pytest.raises(ValueError):
        await store.add([[1.0, 2.0]], [{"foo": "bar"}])


@pytest.mark.asyncio
async def test_search_dimension_mismatch_raises() -> None:
    store = object.__new__(RedisVectorStore)
    store._dim = 3
    store._client = DummyClient()  # type: ignore[attr-defined]
    store._index = "idx"
    with pytest.raises(ValueError):
        await store.search([1.0, 2.0], k=5)
