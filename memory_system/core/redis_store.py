from __future__ import annotations

import uuid
from array import array
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from memory_system.utils.loop import get_loop

from .vector_store import AbstractVectorStore


class RedisVectorStore(AbstractVectorStore):
    """
    Vector store backed by Redis with RediSearch.

    The store is initialised with a fixed vector dimensionality (``dim``). Both
    :meth:`add` and :meth:`search` validate that supplied vectors match this
    dimensionality and raise :class:`ValueError` when they do not.
    """

    def __init__(self, *, url: str, index_name: str, dim: int, allow_flushdb: bool = False) -> None:
        redis = import_module("redis")
        self._client = redis.Redis.from_url(url)
        self._index = index_name
        self._dim = dim
        self._allow_flushdb = allow_flushdb

        self._create_index_if_missing()

    def _create_index_if_missing(self) -> None:
        redis = import_module("redis")
        try:
            self._client.ft(self._index).info()
        except redis.exceptions.ResponseError:
            field_mod = import_module("redis.commands.search.field")
            vector_field = field_mod.VectorField
            text_field = field_mod.TextField
            self._client.ft(self._index).create_index(
                (
                    text_field("id"),
                    vector_field(
                        "vector",
                        "FLAT",
                        {
                            "TYPE": "FLOAT32",
                            "DIM": self._dim,
                            "DISTANCE_METRIC": "COSINE",
                        },
                    ),
                )
            )

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        if any(len(vec) != self._dim for vec in vectors):
            raise ValueError(f"vector dimensionality mismatch, expected {self._dim}")
        ids = [str(uuid.uuid4()) for _ in vectors]
        loop = get_loop()

        def _add() -> None:
            pipe = self._client.pipeline()
            for _id, vec, meta in zip(ids, vectors, metadata, strict=False):
                data = {"id": _id, "vector": array("f", vec).tobytes()}
                data.update({k: str(v) for k, v in meta.items()})
                pipe.hset(_id, mapping=data)
            pipe.execute()

        await loop.run_in_executor(None, _add)
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        if len(vector) != self._dim:
            raise ValueError(f"vector dimensionality mismatch, expected {self._dim}")
        loop = get_loop()

        def _search() -> list[tuple[str, float]]:
            query = f"*=>[KNN {k} @vector $vec AS score]"
            params = {"vec": array("f", vector).tobytes()}
            res = self._client.ft(self._index).search(query, query_params=params)
            return [(doc.id, float(doc.score)) for doc in getattr(res, "docs", [])]

        return await loop.run_in_executor(None, _search)

    async def delete(self, ids: Sequence[str]) -> None:
        """
        Delete the given IDs from Redis.

        Deletions are batched using a single pipeline execution to minimise
        round-trips to Redis.
        """
        loop = get_loop()

        def _del() -> None:
            pipe = self._client.pipeline()
            pipe.delete(*ids)
            pipe.execute()

        await loop.run_in_executor(None, _del)

    async def reset_index(self) -> None:
        """Safely drop and recreate the RediSearch index."""
        loop = get_loop()

        def _work() -> None:
            try:
                self._client.ft(self._index).dropindex(delete_documents=True)
            except Exception:
                pass
            self._create_index_if_missing()

        await loop.run_in_executor(None, _work)

    async def flush_namespace(self, prefix: str) -> None:
        """Remove all keys that start with ``prefix`` using ``SCAN``/``UNLINK``."""
        loop = get_loop()

        def _scan_unlink() -> None:
            cursor = 0
            pattern = f"{prefix}:*"
            while True:
                cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=1000)
                if keys:
                    self._client.unlink(*keys)
                if cursor == 0:
                    break

        await loop.run_in_executor(None, _scan_unlink)

    async def flush_db(self) -> None:
        """Hard reset by issuing ``FLUSHDB`` and recreating the index."""
        if not self._allow_flushdb:
            raise RuntimeError("FLUSHDB disabled in this environment")

        loop = get_loop()

        def _flush() -> None:
            self._client.flushdb()
            self._create_index_if_missing()

        await loop.run_in_executor(None, _flush)

    async def flush(self) -> None:
        await self.flush_db()

    async def close(self) -> None:
        self._client.close()
