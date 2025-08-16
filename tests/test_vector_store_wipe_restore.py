from pathlib import Path

import pytest

try:  # real numpy is optional
    import numpy as np

    _HAS_REAL_NUMPY = hasattr(np.array([0]), "ndim")
except Exception:  # pragma: no cover - fallback when numpy missing
    from tests._stubs import numpy as np

    _HAS_REAL_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_REAL_NUMPY, reason="numpy runtime required")

from memory_system.core.vector_store import VectorStore


@pytest.mark.asyncio
async def test_wipe_is_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "vec"
    with VectorStore(base, dim=4) as store:
        store.add_vector("a", np.zeros(4, dtype=np.float32))
        assert set(store.list_ids()) == {"a"}
        store.wipe()
        assert store.list_ids() == []
        store.add_vector("a", np.zeros(4, dtype=np.float32))
        assert set(store.list_ids()) == {"a"}


@pytest.mark.asyncio
async def test_restore_from_backup_if_enabled(tmp_path: Path) -> None:
    base = tmp_path / "vec"
    with VectorStore(base, dim=4) as store:
        store.add_vector("a", np.zeros(4, dtype=np.float32))
        await store.replicate()
        store.wipe()
        assert store.list_ids() == []

    # simulate backup-enabled restoration
    VectorStore.restore_latest_backup(base)
    with VectorStore(base, dim=4) as restored:
        assert set(restored.list_ids()) == {"a"}
