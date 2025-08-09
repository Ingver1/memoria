from __future__ import annotations

import asyncio
import uuid
from functools import partial
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from .vector_store import AbstractVectorStore


class QdrantVectorStore(AbstractVectorStore):
    """Vector store backed by a Qdrant collection."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dim: int,
        api_key: str | None = None,
    ) -> None:
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._dim = dim
        if not self._client.collection_exists(collection_name=collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    async def add(self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [models.PointStruct(id=ids[i], vector=vectors[i], payload=metadata[i]) for i in range(len(vectors))]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._client.upsert, collection_name=self._collection, points=points))
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                self._client.search,
                collection_name=self._collection,
                query_vector=vector,
                limit=k,
            ),
        )
        return [(str(r.id), float(r.score)) for r in result]

    async def delete(self, ids: Sequence[str]) -> None:
        selector = models.PointIdsList(points=list(ids))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(
                self._client.delete,
                collection_name=self._collection,
                points_selector=selector,
            ),
        )

    async def flush(self) -> None:  # pragma: no cover - Qdrant persists automatically
        return None

    async def close(self) -> None:
        await self.flush()
        self._client.close()
