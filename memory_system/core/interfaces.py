from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from memory_system.core.store import Memory


class VectorStore(ABC):
    """Abstract interface for vector backends."""

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text") -> None:
        """Add vectors for the given ``ids``."""

    @abstractmethod
    def search(
        self,
        vector: NDArray[np.float32],
        *,
        k: int = 5,
        modality: str = "text",
        ef_search: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Search for nearest neighbours."""

    @abstractmethod
    def update(self, ids: Sequence[str], vectors: NDArray[np.float32], *, modality: str = "text") -> None:
        """Update existing vectors."""

    @abstractmethod
    def delete(self, ids: Sequence[str], *, modality: str = "text") -> None:
        """Remove vectors for ``ids``."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to ``path``."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index data from ``path``."""

    @abstractmethod
    def stats(self, modality: str | None = None) -> Any:
        """Return backend statistics."""

    @property
    @abstractmethod
    def ef_search(self) -> int:
        """Return current HNSW ``ef_search`` parameter."""


class MetaStore(ABC):
    """Abstract interface for metadata storage."""

    @abstractmethod
    async def add(self, mem: Memory) -> None:
        """Add a single memory entry."""

    @abstractmethod
    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None:
        """Insert multiple memories."""

    @abstractmethod
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

    @abstractmethod
    async def update(self, memory_id: str, **kwargs: Any) -> Memory:
        """Update a memory and return the new value."""

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """Delete memory by ``memory_id``."""

    @abstractmethod
    def add_commit_hook(self, hook: Any) -> None:
        """Register a commit hook."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close the store."""

    @abstractmethod
    async def ping(self) -> None:
        """Ping the backend."""
