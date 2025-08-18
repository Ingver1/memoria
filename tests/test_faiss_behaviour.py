import pytest

pytestmark = pytest.mark.needs_faiss

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from memory_system.core.index import FaissHNSWIndex


def test_normalize_l2_idempotent() -> None:
    rng = np.random.default_rng(42)
    vecs = rng.random((10, 16), dtype=np.float32)
    norm1 = vecs.copy()
    faiss.normalize_L2(norm1)
    norm2 = norm1.copy()
    faiss.normalize_L2(norm2)
    assert np.allclose(norm1, norm2)
    assert np.allclose(np.linalg.norm(norm1, axis=1), 1.0)


def test_ef_search_recall_monotonic() -> None:
    dim = 32
    rng = np.random.default_rng(0)
    xb = rng.random((500, dim), dtype=np.float32)
    xq = rng.random((20, dim), dtype=np.float32)
    ids = [f"id{i}" for i in range(len(xb))]
    index = FaissHNSWIndex(dim=dim, space="l2")
    index.add_vectors(ids, xb)

    k = 5
    gt = faiss.IndexFlatL2(dim)
    gt.add(xb)
    _, I_gt = gt.search(xq, k)
    gt_ids = [[ids[j] for j in row] for row in I_gt]

    def recall_at(ef: int) -> float:
        total = 0.0
        for q_vec, truth in zip(xq, gt_ids, strict=False):
            found, _ = index.search(q_vec, k=k, ef_search=ef)
            total += len(set(found) & set(truth)) / k
        return total / len(xq)

    recalls = [recall_at(ef) for ef in (8, 32, 128)]
    assert recalls[0] <= recalls[1] <= recalls[2]


def test_save_load_preserves_results(tmp_path) -> None:
    dim = 16
    rng = np.random.default_rng(123)
    xb = rng.random((100, dim), dtype=np.float32)
    ids = [f"id{i}" for i in range(len(xb))]
    index = FaissHNSWIndex(dim=dim)
    index.add_vectors(ids, xb)
    query = xb[1]
    before = index.search(query, k=5)
    path = tmp_path / "index.faiss"
    index.save(path.as_posix())
    new_index = FaissHNSWIndex(dim=dim)
    new_index.load(path.as_posix())
    after = new_index.search(query, k=5)
    assert before[0] == after[0]
    np.testing.assert_allclose(before[1], after[1])
