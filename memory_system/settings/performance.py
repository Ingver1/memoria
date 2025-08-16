from __future__ import annotations

from .base import BaseModel, ConfigDict, PositiveInt, model_validator


class PerformanceConfig(BaseModel):
    """Tuning knobs for throughput and caching."""

    max_workers: int = 4
    cache_size: PositiveInt = 1_000
    cache_ttl_seconds: PositiveInt = 300
    rebuild_interval_seconds: PositiveInt = 3_600
    queue_max_size: PositiveInt = 1_000
    deep_dedup: bool = True
    async_timeout: PositiveInt = 5

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _workers_range(self) -> PerformanceConfig:
        if self.max_workers < 1 or self.max_workers > 32:
            raise ValueError("max_workers must be between 1 and 32")
        return self


__all__ = ["PerformanceConfig"]
