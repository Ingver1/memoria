import pytest

from memory_system.core.duckdb_store import DuckDBVectorStore


@pytest.mark.asyncio
async def test_duckdb_search_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    duckdb = pytest.importorskip("duckdb")  # skip if duckdb isn't installed
    store = DuckDBVectorStore(path=":memory:", dim=2)
    ids = await store.add([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [{}, {}, {}])

    executed: list[str] = []
    original_execute = store._conn.execute  # type: ignore[attr-defined]

    def spy(sql: str, *args, **kwargs):
        executed.append(sql)
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store._conn, "execute", spy)  # type: ignore[arg-type]

    results = await store.search([1.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0][0] == ids[0]
    assert results[0][1] == pytest.approx(0.0)
    # Second result should be the vector [1.0, 1.0]
    assert results[1][0] == ids[2]

    # Ensure SQL query applied LIMIT for performance
    assert any("LIMIT" in sql.upper() for sql in executed)
