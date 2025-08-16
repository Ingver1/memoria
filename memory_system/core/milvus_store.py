from __future__ import annotations

import uuid
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from memory_system.utils.loop import get_loop

from .vector_store import AbstractVectorStore


class MilvusVectorStore(AbstractVectorStore):
    """Vector store backed by a Milvus collection."""

    def __init__(
        self,
        *,
        uri: str,
        collection: str,
        dim: int,
        token: str | None = None,
    ) -> None:
        pymilvus = import_module("pymilvus")
        self._pymilvus = pymilvus
        pymilvus.connections.connect(alias="default", uri=uri, token=token)
        self._collection_name = collection
        self._dim = dim
        if collection not in pymilvus.list_collections():
            field_id = pymilvus.FieldSchema(
                name="id", dtype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=36
            )
            field_vec = pymilvus.FieldSchema(
                name="vector", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dim
            )
            field_meta = pymilvus.FieldSchema(name="payload", dtype=pymilvus.DataType.JSON)
            schema = pymilvus.CollectionSchema(fields=[field_id, field_vec, field_meta])
            pymilvus.Collection(name=collection, schema=schema)
        self._collection = pymilvus.Collection(collection)

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        loop = get_loop()

        def _insert() -> None:
            self._collection.insert([ids, vectors, list(metadata)])

        await loop.run_in_executor(None, _insert)
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        loop = get_loop()

        def _search() -> list[tuple[str, float]]:
            res = self._collection.search(
                data=[vector],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=k,
                output_fields=["id"],
            )
            hits = res[0] if res else []
            return [(hit.id, float(hit.distance)) for hit in hits]

        return await loop.run_in_executor(None, _search)

    async def delete(self, ids: Sequence[str]) -> None:
        loop = get_loop()

        def _del() -> None:
            expr = "id in [" + ",".join(f'"{i}"' for i in ids) + "]"
            self._collection.delete(expr)

        await loop.run_in_executor(None, _del)

    async def flush(self) -> None:
        loop = get_loop()
        await loop.run_in_executor(None, self._collection.flush)

    async def close(self) -> None:
        await self.flush()
        self._pymilvus.connections.disconnect(alias="default")
