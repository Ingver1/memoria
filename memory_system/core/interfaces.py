"""Protocol interfaces for vector and metadata stores."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from memory_system.core.store import Memory

Float32Array: TypeAlias = NDArray[np.float32]


@runtime_checkable
class VectorStore(Protocol):
    """Protocol defining the expected vector backend behaviour."""

    def add(self, ids: Sequence[str], vectors: Float32Array, *, modality: str = "text") -> None:
        """Add vectors for the given ``ids``."""

    def search(
        self,
        vector: Float32Array,
        *,
        k: int = 5,
        modality: str = "text",
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Search for nearest neighbours."""

    def update(self, ids: Sequence[str], vectors: Float32Array, *, modality: str = "text") -> None:
        """Update existing vectors."""

    def delete(self, ids: Sequence[str], *, modality: str = "text") -> None:
        """Remove vectors for ``ids``."""

    def rebuild(self, modality: str, vectors: Float32Array, ids: Sequence[str]) -> None:
        """Rebuild index for ``modality`` using ``vectors`` and ``ids``."""

    def save(self, path: str) -> None:
        """Persist the index to ``path``."""

    def load(self, path: str) -> None:
        """Load index data from ``path``."""

    def stats(self, modality: str | None = None) -> dict[str, Any]:
        """Return backend statistics."""

    @property
    def ef_search(self) -> int:
        """Return current HNSW ``ef_search`` parameter."""

    def set_ef_search(self, value: int) -> None:
        """Update the ``ef_search`` parameter if supported."""


@runtime_checkable
class MetaStore(Protocol):
    """Protocol for metadata storage backends."""

    async def add(self, mem: Memory) -> None:
        """Add a single memory entry."""

    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None:
        """Insert multiple memories."""

    async def search(
        self,
        text_query: str | None = None,
        *,
        metadata_filters: dict[str, Any] | None = None,
        limit: int = 20,
        level: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Search stored memories."""

    async def update(self, memory_id: str, **kwargs: Any) -> Memory:  # noqa: ANN401
        """Update a memory and return the new value."""

    async def delete(self, memory_id: str) -> None:
        """Delete memory by ``memory_id``."""

    async def _acquire(self, *, write: bool = False) -> Any:
        """Acquire a low-level database connection."""

    async def _release(self, conn: Any) -> None:
        """Release a previously acquired connection."""

    _doc_count: int

    def _update_df_cache(self, text: str, delta: int) -> None:
        """Update token document-frequency cache."""

    async def _run_commit_hooks(self) -> None:
        """Execute registered commit hooks."""

    def add_commit_hook(self, hook: Any) -> None:  # noqa: ANN401
        """Register a commit hook."""

    async def aclose(self) -> None:
        """Close the store."""

    async def ping(self) -> None:
        """Ping the backend."""


@runtime_checkable
class VectorIndexMaintenance(Protocol):
    """Protocol for vector index maintenance operations."""

    _id_map: Mapping[Any, str]

    def list_ids(self) -> Sequence[str] | Awaitable[Sequence[str]]:
        """Return all IDs stored in the index."""

    def add_vectors(self, ids: Sequence[str], vectors: Float32Array) -> None:
        """Insert vectors for ``ids`` into the index."""

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove ``ids`` from the index asynchronously."""

    def remove_ids(self, ids: Sequence[str]) -> None:
        """Remove ``ids`` from the index synchronously."""
