import pytest

pytestmark = [pytest.mark.needs_faiss, pytest.mark.needs_numpy]

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from memory_system.core.faiss_vector_store import FaissVectorStore
from memory_system.settings import UnifiedSettings


def test_autotune_on_initialisation(tmp_path):
    path = tmp_path / "index.faiss"

    settings = UnifiedSettings(
        database={"vec_path": path},
        model={"vector_dim": 16},
        faiss={"autotune": False},
    )
    store = FaissVectorStore(settings)
    rng = np.random.default_rng(0)
    xb = rng.random((200, 16), dtype=np.float32)
    ids = [f"id{i}" for i in range(len(xb))]
    store.add(ids, xb)
    store.save()

    settings2 = UnifiedSettings(
        database={"vec_path": path},
        model={"vector_dim": 16},
        faiss={"autotune": True},
    )
    store2 = FaissVectorStore(settings2)
    assert store2._autotune is False
    assert settings2.faiss.autotune is False
    assert settings2.faiss.ef_search is not None
    assert settings2.faiss.ef_construction is not None
