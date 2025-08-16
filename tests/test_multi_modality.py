import pytest

np = pytest.importorskip("numpy")

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.index import MultiModalFaissIndex
from memory_system.settings import UnifiedSettings


def test_multi_modal_index_isolation(tmp_path):
    idx = MultiModalFaissIndex({"text": 2, "image": 3})
    idx.add_vectors("text", ["t1"], np.array([[0.1, 0.2]], dtype=np.float32))
    idx.add_vectors("image", ["i1"], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    ids_text, _ = idx.search("text", np.array([0.1, 0.2], dtype=np.float32), k=1)
    ids_img, _ = idx.search("image", np.array([0.1, 0.2, 0.3], dtype=np.float32), k=1)
    assert ids_text == ["t1"]
    assert ids_img == ["i1"]


def test_multi_modal_auto_tune() -> None:
    idx = MultiModalFaissIndex({"text": 4, "image": 3})
    samples = {
        "text": np.random.rand(10, 4).astype(np.float32),
        "image": np.random.rand(10, 3).astype(np.float32),
    }
    res = idx.auto_tune(samples)
    assert set(res) == {"text", "image"}
    for vals in res.values():
        assert all(isinstance(v, int) for v in vals)


def test_multi_modal_save_load(tmp_path):
    # Skip persistence round-trip under fallback index implementation
    try:
        import memory_system.core._faiss_mod as _fm

        if _fm._faiss_mod.__name__.endswith("_faiss_hnsw_fallback"):
            pytest.skip("Skipping save/load with fallback FAISS implementation")
    except Exception:
        pass
    idx = MultiModalFaissIndex(
        {
            "text": {"dim": 2},
            "image": {"dim": 3, "index_type": "IVF", "ivf_nlist": 1},
        }
    )
    idx.add_vectors("text", ["t1"], np.array([[0.1, 0.2]], dtype=np.float32))
    idx.add_vectors("image", ["i1"], np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    path = tmp_path / "mmindex"
    idx.save(str(path))

    loaded = MultiModalFaissIndex()
    loaded.load(str(path))
    ids_text, _ = loaded.search("text", np.array([0.1, 0.2], dtype=np.float32), k=1)
    ids_img, _ = loaded.search("image", np.array([0.1, 0.2, 0.3], dtype=np.float32), k=1)
    assert ids_text == ["t1"]
    assert ids_img == ["i1"]
    assert loaded._indices["image"].index_type == "IVF"


@pytest.mark.asyncio
async def test_enhanced_store_multi_modality(tmp_path):
    settings = UnifiedSettings.for_testing()
    model = settings.model.model_copy(
        update={
            "modalities": ["text", "image"],
            "vector_dims": {"text": settings.model.vector_dim, "image": 3},
        }
    )
    object.__setattr__(settings, "model", model)
    object.__setattr__(settings.database, "vec_path", tmp_path / "memory.vectors")

    async with EnhancedMemoryStore(settings) as store:
        await store.add_memory(
            text="hello", embedding=[0.1] * settings.model.vector_dims["text"], modality="text"
        )
        await store.add_memory(text="picture", embedding=[0.1, 0.2, 0.3], modality="image")

        res_text = await store.semantic_search(
            vector=[0.1] * settings.model.vector_dims["text"], modality="text"
        )
        res_img = await store.semantic_search(vector=[0.1, 0.2, 0.3], modality="image")
        assert len(res_text) == 1 and res_text[0].text == "hello"
        assert len(res_img) == 1 and res_img[0].text == "picture"


@pytest.mark.asyncio
async def test_enhanced_store_multi_modality_batch(tmp_path):
    settings = UnifiedSettings.for_testing()
    model = settings.model.model_copy(
        update={
            "modalities": ["text", "image"],
            "vector_dims": {"text": settings.model.vector_dim, "image": 3},
        }
    )
    object.__setattr__(settings, "model", model)
    object.__setattr__(settings.database, "vec_path", tmp_path / "memory.vectors")

    async with EnhancedMemoryStore(settings) as store:
        await store.add_memories_batch(
            [
                {
                    "text": "hello",
                    "embedding": [0.1] * settings.model.vector_dims["text"],
                    "modality": "text",
                },
                {
                    "text": "picture",
                    "embedding": [0.1, 0.2, 0.3],
                    "modality": "image",
                },
            ]
        )

        res_text = await store.semantic_search(
            vector=[0.1] * settings.model.vector_dims["text"], modality="text"
        )
        res_img = await store.semantic_search(vector=[0.1, 0.2, 0.3], modality="image")
        assert len(res_text) == 1 and res_text[0].text == "hello"
        assert len(res_img) == 1 and res_img[0].text == "picture"
