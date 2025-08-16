from __future__ import annotations

import uuid
from collections.abc import Sequence
from importlib import import_module
from typing import Any

from memory_system.utils.loop import get_loop

from .vector_store import AbstractVectorStore


class WeaviateVectorStore(AbstractVectorStore):
    """Vector store backed by a Weaviate instance."""

    def __init__(
        self,
        *,
        url: str,
        class_name: str,
        dim: int,
        api_key: str | None = None,
    ) -> None:
        weaviate = import_module("weaviate")
        auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None
        self._client = weaviate.Client(url=url, auth_client_secret=auth)
        self._class = class_name
        self._dim = dim
        if not self._client.schema.contains({"class": class_name}):
            schema = {"class": class_name, "vectorizer": "none", "properties": []}
            self._client.schema.create_class(schema)

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")
        ids = [str(uuid.uuid4()) for _ in vectors]
        loop = get_loop()

        def _add() -> None:
            with self._client.batch as batch:
                for _id, vec, meta in zip(ids, vectors, metadata, strict=False):
                    batch.add_data_object(meta, self._class, uuid=_id, vector=vec)

        await loop.run_in_executor(None, _add)
        return ids

    async def search(self, vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        loop = get_loop()

        def _search() -> list[tuple[str, float]]:
            res = (
                self._client.query.get(self._class, [])
                .with_near_vector({"vector": vector})
                .with_additional(["id", "distance"])
                .with_limit(k)
                .do()
            )
            data = res.get("data", {}).get("Get", {}).get(self._class, [])
            return [
                (item["_additional"]["id"], float(item["_additional"]["distance"])) for item in data
            ]

        return await loop.run_in_executor(None, _search)

    async def delete(self, ids: Sequence[str]) -> None:
        loop = get_loop()

        def _del() -> None:
            for _id in ids:
                self._client.data_object.delete(_id, class_name=self._class)

        await loop.run_in_executor(None, _del)

    async def flush(self) -> None:  # Weaviate persists automatically
        return None

    async def close(self) -> None:
        await self.flush()
