"""
Lightweight wrapper around the ``sqlite-vec`` extension.

This module provides :class:`SQLiteVecStore` which attempts to use the
``sqlite-vec`` extension for vector search.  When the extension is not
available the implementation falls back to a simple in-memory dictionary
based store so the rest of the system can continue operating.
"""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Sequence
from typing import Any

from .vector_store import register_vector_store


@register_vector_store("sqlite_vec")
class SQLiteVecStore:
    """Small vector store backed by ``sqlite-vec`` when available."""

    def __init__(self, *, path: str, dim: int) -> None:
        self._dim = dim
        self._conn: sqlite3.Connection | None = None
        self._mem: dict[str, tuple[list[float], dict[str, Any]]] = {}

        # Attempt to create a virtual table using the sqlite-vec extension.
        try:
            conn = sqlite3.connect(path)
            try:
                conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(dim={dim})")
            except Exception:
                conn.close()
                raise
            self._conn = conn
        except Exception:  # pragma: no cover - extension not available
            self._conn = None

    # ------------------------------------------------------------------
    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        if self._conn is None:
            for _id, vec, meta in zip(ids, vectors, metadata, strict=False):
                self._mem[_id] = (list(map(float, vec)), dict(meta))
            return ids

        records = [
            (ids[i], "[" + ",".join(str(v) for v in vectors[i]) + "]", metadata[i])
            for i in range(len(vectors))
        ]
        self._conn.executemany("INSERT INTO vec(rowid, value, metadata) VALUES (?, ?, ?)", records)
        self._conn.commit()
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        if self._conn is None:
            # simple linear scan over in-memory dict
            results: list[tuple[str, float]] = []
            for _id, (vec, _meta) in self._mem.items():
                dist = math.sqrt(sum((vector[i] - vec[i]) ** 2 for i in range(self._dim)))
                results.append((_id, dist))
            results.sort(key=lambda x: x[1])
            return results[:k]

        rows = self._conn.execute(
            "SELECT rowid, distance FROM vec WHERE value MATCH ? ORDER BY distance LIMIT ?",
            ("[" + ",".join(str(v) for v in vector) + "]", k),
        ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    async def delete(self, ids: Sequence[str]) -> None:
        if self._conn is None:
            for _id in ids:
                self._mem.pop(_id, None)
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM vec WHERE rowid IN ({placeholders})", list(ids))
        self._conn.commit()

    async def flush(self) -> None:
        if self._conn is not None:
            self._conn.commit()

    async def close(self) -> None:
        await self.flush()
        if self._conn is not None:
            self._conn.close()
