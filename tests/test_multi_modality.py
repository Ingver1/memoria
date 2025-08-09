import pytest

np = pytest.importorskip("numpy")

from memory_system.config.settings import UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.index import MultiModalFaissIndex


def test_multi_modal_index_isolation(tmp_path):
    idx = MultiModalFaissIndex({"text": 2, "image": 3})
    idx.add_vectors("text", ["t1"], np.array([[0.1, 0.2]], dtype=np.float32))
    idx.add_vectors("image", ["i1"], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    ids_text, _ = idx.search("text", np.array([0.1, 0.2], dtype=np.float32), k=1)
    ids_img, _ = idx.search("image", np.array([0.1, 0.2, 0.3], dtype=np.float32), k=1)
    assert ids_text == ["t1"]
    assert ids_img == ["i1"]


@pytest.mark.asyncio
async def test_enhanced_store_multi_modality(tmp_path):
    settings = UnifiedSettings.for_testing()
    model = settings.model.model_copy(
        update={"modalities": ["text", "image"], "vector_dims": {"text": settings.model.vector_dim, "image": 3}}
    )
    object.__setattr__(settings, "model", model)
    object.__setattr__(settings.database, "vec_path", tmp_path / "memory.vectors")

    store = EnhancedMemoryStore(settings)
    await store.start()
    await store.add_memory(text="hello", embedding=[0.1] * settings.model.vector_dims["text"], modality="text")
    await store.add_memory(text="picture", embedding=[0.1, 0.2, 0.3], modality="image")

    res_text = await store.semantic_search(vector=[0.1] * settings.model.vector_dims["text"], modality="text")
    res_img = await store.semantic_search(vector=[0.1, 0.2, 0.3], modality="image")
    assert len(res_text) == 1 and res_text[0].text == "hello"
    assert len(res_img) == 1 and res_img[0].text == "picture"
    await store.close()
