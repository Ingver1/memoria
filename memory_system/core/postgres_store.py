from __future__ import annotations

"""Asynchronous PostgreSQL-backed memory and vector stores."""

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

try:  # pragma: no cover - psycopg optional
    import psycopg
    from psycopg.types.json import Json
except Exception:  # pragma: no cover - psycopg optional
    psycopg = None
    Json = None

try:  # pragma: no cover - pgvector optional
    from pgvector.psycopg import register_vector
except Exception:  # pragma: no cover - pgvector optional
    register_vector = None

from .store import Memory
from .vector_store import register_vector_store


class PostgresMemoryStore:
    """Lightweight memory store persisting data in PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        if psycopg is None:  # pragma: no cover - defensive
            raise RuntimeError("psycopg is required for PostgresMemoryStore")
        self._dsn = dsn
        self._pool: psycopg.AsyncConnectionPool | None = None

    async def initialise(self) -> None:
        """Initialise connection pool and ensure schema exists."""
        assert psycopg is not None
        if self._pool is None:
            self._pool = psycopg.AsyncConnectionPool(self._dsn, open=False)
            await self._pool.open()
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY,
                    text TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    valid_from TIMESTAMPTZ,
                    valid_to TIMESTAMPTZ,
                    tx_from TIMESTAMPTZ,
                    tx_to TIMESTAMPTZ,
                    importance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    valence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    emotional_intensity DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    metadata JSONB,
                    level INT NOT NULL DEFAULT 0,
                    episode_id TEXT,
                    modality TEXT NOT NULL DEFAULT 'text',
                    connections JSONB,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    pinned BOOLEAN NOT NULL DEFAULT FALSE,
                    ttl_seconds INT,
                    last_used TIMESTAMPTZ,
                    success_score DOUBLE PRECISION,
                    decay DOUBLE PRECISION
                )
                """
            )

    async def aclose(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ------------------------------------------------------------------
    async def add_memory(self, mem: Memory) -> None:
        """Insert ``mem`` into the database."""
        assert self._pool is not None
        data = asdict(mem)
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memories (
                    id, text, created_at, valid_from, valid_to, tx_from, tx_to,
                    importance, valence, emotional_intensity, metadata, level,
                    episode_id, modality, connections, memory_type, pinned,
                    ttl_seconds, last_used, success_score, decay
                ) VALUES (
                    %(id)s, %(text)s, %(created_at)s, %(valid_from)s, %(valid_to)s,
                    %(tx_from)s, %(tx_to)s, %(importance)s, %(valence)s,
                    %(emotional_intensity)s, %(metadata)s, %(level)s,
                    %(episode_id)s, %(modality)s, %(connections)s, %(memory_type)s,
                    %(pinned)s, %(ttl_seconds)s, %(last_used)s, %(success_score)s,
                    %(decay)s
                )
                ON CONFLICT (id) DO NOTHING
                """,
                data,
            )

    async def get_memory(self, mem_id: str) -> Memory | None:
        """Fetch a memory by ``mem_id``."""
        assert self._pool is not None
        async with self._pool.connection() as conn:
            cur = await conn.execute("SELECT * FROM memories WHERE id = %(id)s", {"id": mem_id})
            row = await cur.fetchone()
            if not row:
                return None
            return Memory(**row)

    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None:
        """Insert multiple memories in batches."""
        for i in range(0, len(memories), batch_size):
            batch = memories[i : i + batch_size]
            await asyncio.gather(*(self.add_memory(m) for m in batch))

    # ------------------------------------------------------------------
    async def search_iter(self, *args: Any) -> Any:
        """Placeholder for compatibility with ``MemoryStoreProtocol``."""
        raise NotImplementedError("search_iter is not implemented for Postgres store")


@register_vector_store("postgres")
class PostgresVectorStore:
    """Vector store backed by PostgreSQL with the ``pgvector`` extension."""

    def __init__(self, *, dsn: str, table: str, dim: int) -> None:
        if psycopg is None or register_vector is None or Json is None:
            msg = "psycopg and pgvector are required for PostgresVectorStore"
            raise RuntimeError(msg)
        self._dsn = dsn
        self._table = table
        self._dim = dim
        self._pool: psycopg.AsyncConnectionPool | None = None

    async def _ensure_pool(self) -> psycopg.AsyncConnectionPool:
        assert psycopg is not None and register_vector is not None
        if self._pool is None:
            self._pool = psycopg.AsyncConnectionPool(self._dsn, open=False)
            await self._pool.open()
            async with self._pool.connection() as conn:
                await register_vector(conn)
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id UUID PRIMARY KEY,
                        embedding vector({self._dim}),
                        metadata JSONB
                    )
                    """
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
                    f"ON {self._table} USING ivfflat (embedding vector_cosine_ops)"
                )
        return self._pool

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        pool = await self._ensure_pool()
        records = [(ids[i], vectors[i], Json(metadata[i])) for i in range(len(vectors))]
        async with pool.connection() as conn:
            await register_vector(conn)
            await conn.executemany(
                f"""
                INSERT INTO {self._table} (id, embedding, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                records,
            )
        return ids

    async def search(
        self,
        vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        pool = await self._ensure_pool()
        where_clauses: list[str] = []
        params: list[Any] = []
        if filters:
            for key, value in filters.items():
                where_clauses.append("metadata @> %s")
                params.append(Json({key: value}))
        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        query = (
            f"SELECT id, embedding <=> %s AS dist FROM {self._table}"
            f"{where_sql} ORDER BY embedding <=> %s LIMIT %s"
        )
        params.extend([vector, vector, k])
        async with pool.connection() as conn:
            await register_vector(conn)
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            await conn.execute(
                f"DELETE FROM {self._table} WHERE id = ANY(%s)",
                (list(ids),),
            )

    async def flush(self) -> None:  # pragma: no cover - autocommit
        return None

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


__all__ = ["PostgresMemoryStore", "PostgresVectorStore"]
