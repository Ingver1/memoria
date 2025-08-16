import pytest

pytestmark = pytest.mark.needs_faiss

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from memory_system.core.faiss_vector_store import FaissVectorStore
from memory_system.settings import UnifiedSettings


def test_set_ef_search_persistence() -> None:
    settings = UnifiedSettings.for_testing()
    store = FaissVectorStore(settings)
    vec = np.random.rand(settings.model.vector_dim).astype(np.float32)
    store.add(["id"], vec.reshape(1, -1))
    start = store.ef_search
    new_val = start * 2
    store.set_ef_search(new_val)
    assert store.ef_search == new_val
    store.search(vec, k=1)
    assert store.ef_search == new_val
    store.set_ef_search(start)
    store.search(vec, k=1)
    assert store.ef_search == start
