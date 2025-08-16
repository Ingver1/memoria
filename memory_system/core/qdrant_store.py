from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient as _QdrantClient, models as _models
except ImportError:  # pragma: no cover - qdrant-client optional
    _QdrantClient = None
    _models = None
QdrantClient = _QdrantClient
models = _models

from memory_system.utils.loop import get_loop

from .vector_store import register_vector_store

log = logging.getLogger(__name__)


@register_vector_store("qdrant")
class QdrantVectorStore:
    """Vector store backed by a Qdrant collection."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        dim: int,
        api_key: str | None = None,
    ) -> None:
        if QdrantClient is None or models is None:
            msg = "qdrant-client is required for QdrantVectorStore"
            raise ImportError(msg)

        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection = collection
        self._dim = dim
        self._indexed_fields: set[str] = set()
        if not self._client.collection_exists(collection_name=collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        points = [
            models.PointStruct(id=ids[i], vector=vectors[i], payload=metadata[i])
            for i in range(len(vectors))
        ]
        loop = get_loop()
        await loop.run_in_executor(
            None, partial(self._client.upsert, collection_name=self._collection, points=points)
        )
        return ids

    def _ensure_payload_index(self, field: str, value: Any) -> None:
        if field in self._indexed_fields:
            return
        if isinstance(value, bool):
            schema = models.PayloadSchemaType.BOOL
        elif isinstance(value, int):
            schema = models.PayloadSchemaType.INTEGER
        elif isinstance(value, float):
            schema = models.PayloadSchemaType.FLOAT
        else:
            schema = models.PayloadSchemaType.KEYWORD
        try:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=schema,
            )
            self._indexed_fields.add(field)
        except Exception:  # pragma: no cover - best effort
            log.debug("payload index creation failed", exc_info=True)

    async def ping(self) -> bool:
        """Return ``True`` if the collection is reachable."""
        loop = get_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.get_collection, collection_name=self._collection),
            )
            return True
        except Exception:  # pragma: no cover - network failure
            return False

    async def search(
        self,
        vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        loop = get_loop()
        query_filter = None
        if filters:
            for key, val in filters.items():
                await loop.run_in_executor(None, partial(self._ensure_payload_index, key, val))
            conditions = [
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
            query_filter = models.Filter(must=conditions)
        result = await loop.run_in_executor(
            None,
            partial(
                self._client.search,
                collection_name=self._collection,
                query_vector=vector,
                query_filter=query_filter,
                limit=k,
            ),
        )
        return [(str(r.id), float(r.score)) for r in result]

    async def list_ids(self) -> list[str]:
        """Return all point IDs from the collection."""
        loop = get_loop()
        ids: list[str] = []
        offset: Any | None = None
        while True:
            res = await loop.run_in_executor(
                None,
                partial(
                    self._client.scroll,
                    collection_name=self._collection,
                    offset=offset,
                    with_vectors=False,
                    with_payload=False,
                    limit=1_000,
                ),
            )
            res = cast("Any", res)
            ids.extend(str(p.id) for p in res.points)
            offset = res.next_page_offset
            if offset is None:
                break
        return ids

    async def delete(self, ids: Sequence[str]) -> None:
        selector = models.PointIdsList(points=list(ids))
        loop = get_loop()
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
