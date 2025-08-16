import pytest

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["batch", "streaming"])
async def test_prepare_entry_shared_helper(tmp_path, method):
    settings = UnifiedSettings.for_testing()
    settings.security = settings.security.model_copy(update={"filter_pii": True})
    object.__setattr__(settings.database, "db_path", tmp_path / "memory.db")
    object.__setattr__(settings.database, "vec_path", tmp_path / "memory.vectors")

    store = EnhancedMemoryStore(settings)
    await store.start()

    item = {"text": "email me at test@example.com", "embedding": [0.1] * settings.model.vector_dim}
    if method == "batch":
        await store.add_memories_batch([item])
    else:
        await store.add_memories_streaming([item])

    records = await store.meta_store.search()
    assert records[0].text == "email me at [EMAIL_REDACTED]"

    res = await store.semantic_search(vector=[0.1] * settings.model.vector_dim)
    assert res and res[0].text == "email me at [EMAIL_REDACTED]"
    await store.close()
