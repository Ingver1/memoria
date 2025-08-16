from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .base import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class DatabaseConfig(BaseModel):
    """Paths and connection limits for the storage backend."""

    url: str | None = None
    engine: Literal["sqlite", "postgres"] = "sqlite"
    postgres_dsn: str | None = None
    db_path: Path = Path("data/memory.db")
    vec_path: Path = Path("data/memory.vectors")
    cache_path: Path = Path("data/memory.cache")
    connection_pool_size: PositiveInt = 10
    backend: str = "faiss"
    backend_config: dict[str, Any] = Field(default_factory=dict)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "memory"
    wal: bool = True
    synchronous: Literal["OFF", "NORMAL", "FULL", "EXTRA"] = "NORMAL"
    wal_checkpoint_interval: float = 60.0
    wal_checkpoint_writes: PositiveInt = 1_000
    page_size: PositiveInt | None = None
    cache_size: int | None = None
    mmap_size: int | None = None
    busy_timeout: int = 5_000

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _coerce_path(self) -> DatabaseConfig:
        object.__setattr__(
            self, "db_path", self.db_path if isinstance(self.db_path, Path) else Path(self.db_path)
        )
        object.__setattr__(
            self,
            "vec_path",
            self.vec_path if isinstance(self.vec_path, Path) else Path(self.vec_path),
        )
        object.__setattr__(
            self,
            "cache_path",
            self.cache_path if isinstance(self.cache_path, Path) else Path(self.cache_path),
        )
        return self

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if self.model_config.get("frozen") and name in self.__dict__:
            raise ValueError("DatabaseConfig is immutable")
        super().__setattr__(name, value)


__all__ = ["DatabaseConfig"]
