from pathlib import Path

import pytest

from memory_system.core.store import SQLiteMemoryStore
from memory_system.core.top_n_by_score_sql import build_top_n_by_score_sql
from memory_system.unified_memory import ListBestWeights


def test_build_top_n_by_score_sql_rejects_invalid_key():
    weights = ListBestWeights(1, 1, 1, 1)
    with pytest.raises(ValueError):
        build_top_n_by_score_sql(5, weights, metadata_filter={"role;DROP TABLE": "x"})


@pytest.mark.asyncio
async def test_store_top_n_by_score_rejects_invalid_metadata_key(tmp_path: Path):
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()
    try:
        with pytest.raises(ValueError):
            await store.top_n_by_score(1, metadata_filter={"role;DROP TABLE": "x"})
    finally:
        await store.aclose()
