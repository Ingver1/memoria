import pytest

from memory_system.core.sqlite_vec_store import SQLiteVecStore


@pytest.mark.asyncio
async def test_sqlite_vec_store_fallback() -> None:
    store = SQLiteVecStore(path=":memory:", dim=2)
    # sqlite-vec extension is unlikely to be available in the test env so the
    # store should fall back to its in-memory dictionary implementation.
    ids = await store.add([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [{}, {}, {}])
    results = await store.search([1.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0][0] == ids[0]
    # Second closest vector should be [1.0, 1.0]
    assert results[1][0] == ids[2]
