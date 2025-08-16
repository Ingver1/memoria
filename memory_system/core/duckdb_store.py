from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from typing import Any

try:  # optional duckdb dependency
    import duckdb
except Exception:  # pragma: no cover - optional backend
    duckdb = None

from .vector_store import register_vector_store


@register_vector_store("duckdb")
class DuckDBVectorStore:
    """
    Very small DuckDB-backed vector store.

    When the optional :mod:`duckdb` dependency is unavailable the implementation
    falls back to an in-memory dictionary.  This keeps the class lightweight
    while still exercising the same code paths for testing purposes.
    """

    def __init__(self, *, path: str, dim: int) -> None:
        self._dim = dim
        if duckdb is None:
            self._vectors: dict[str, list[float]] = {}
        else:
            self._conn = duckdb.connect(path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY, vec JSON, meta JSON)"
            )

    async def add(
        self,
        vectors: Sequence[list[float]],
        metadata: Sequence[dict[str, Any]],
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        if duckdb is None:
            for vid, vec in zip(ids, vectors, strict=False):
                self._vectors[vid] = vec
            return ids
        records = [
            (ids[i], json.dumps(vectors[i]), json.dumps(metadata[i])) for i in range(len(vectors))
        ]
        self._conn.executemany("INSERT INTO vectors VALUES (?, ?, ?)", records)
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        def _distance(a: list[float], b: list[float]) -> float:
            return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)) ** 0.5)

        if duckdb is None:
            scores = [(_id, _distance(vec, vector)) for _id, vec in self._vectors.items()]
            scores.sort(key=lambda x: x[1])
            return scores[:k]

        components = [
            f"POWER(CAST(json_extract(vec, '$[{i}]') AS DOUBLE) - ?, 2)" for i in range(self._dim)
        ]
        dist_expr = " + ".join(components)
        sql = f"SELECT id, SQRT({dist_expr}) AS distance FROM vectors ORDER BY distance LIMIT ?"
        params = [*vector, k]
        rows = self._conn.execute(sql, params).fetchall()
        return [(row[0], float(row[1])) for row in rows]

    async def delete(self, ids: Sequence[str]) -> None:
        if duckdb is None:
            for _id in ids:
                self._vectors.pop(_id, None)
        else:
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(f"DELETE FROM vectors WHERE id IN ({placeholders})", list(ids))

    async def flush(self) -> None:  # pragma: no cover - lightweight
        if duckdb is not None:
            self._conn.commit()

    async def close(self) -> None:
        await self.flush()
        if duckdb is not None:
            self._conn.close()
