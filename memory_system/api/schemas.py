# schemas.py — Pydantic models for Unified Memory System API
#
# Version: v{__version__}
"""Centralised data‑contracts used by the REST API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator
from types import SimpleNamespace

from memory_system import __version__

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_API_VERSION: str = "v1"
_SERVICE_VERSION: str = __version__


# ---------------------------------------------------------------------------
# Memory domain models
# ---------------------------------------------------------------------------
class MemoryBase(BaseModel):
    """Fields common to create / update operations."""

    text: str = Field(..., min_length=1, max_length=10_000)
    role: str = Field("user", max_length=32, description="Conversation role label")
    tags: list[str] = Field(default_factory=list, max_length=10)
    valence: float = Field(0.0, description="Emotion polarity")
    emotional_intensity: float = Field(
        default_factory=lambda: _get_dynamics().initial_intensity,
        description="Strength of emotional reaction",
        validation_alias=AliasChoices("arousal", "emotional_intensity"),
        serialization_alias="arousal",
    )
    importance: float = Field(
        0.0,
        description="Subjective importance of the memory",
    )
    modality: str = Field("text", max_length=32)
    language: str | None = Field(None, max_length=32)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.text or len(self.text) > 10_000:
            raise ValueError("text must be between 1 and 10000 characters")
        if len(self.role) > 32:
            raise ValueError("role too long")
        if len(self.tags) > 10:
            raise ValueError("too many tags")
        self.valence = max(-1.0, min(1.0, self.valence))
        self.emotional_intensity = max(0.0, min(1.0, self.emotional_intensity))
        self.importance = max(0.0, min(1.0, self.importance))


class MemoryCreate(MemoryBase):
    """Payload for *create* operation."""

    user_id: str | None = Field(
        None,
        description="Owner identifier (if omitted — resolved from auth context)",
    )


class MemoryUpdate(BaseModel):
    """Payload for partial updates where all fields are optional.

    ``valence``, ``emotional_intensity`` and ``importance`` replace the
    existing values directly, whereas the ``*_delta`` counterparts add to the
    current value.
    """

    text: str | None = Field(default=None, min_length=1, max_length=10_000)
    role: str | None = Field(default=None, max_length=32)
    tags: list[str] | None = Field(default=None, max_length=10)
    valence: float | None = Field(
        default=None,
        description="Set absolute valence value",
    )
    emotional_intensity: float | None = Field(
        default=None,
        validation_alias=AliasChoices("arousal", "emotional_intensity"),
        serialization_alias="arousal",
        description="Set absolute arousal/emotional intensity",
    )
    importance: float | None = Field(
        default=None,
        description="Set absolute importance",
    )
    valence_delta: float | None = Field(
        default=None,
        description="Increment to apply to current valence",
    )
    emotional_intensity_delta: float | None = Field(
        default=None,
        validation_alias=AliasChoices("arousal_delta", "emotional_intensity_delta"),
        serialization_alias="arousal_delta",
        description="Increment to apply to current arousal",
    )
    importance_delta: float | None = Field(
        default=None,
        description="Increment to apply to current importance",
    )

    model_config = {
        "extra": "forbid",
        "validate_default": True,
    }

    @field_validator("valence", "valence_delta")
    @classmethod
    def _clamp_valence(cls, v: float | None) -> float | None:
        if v is None:
            return v
        return max(-1.0, min(1.0, v))

    @field_validator("emotional_intensity", "emotional_intensity_delta", "importance", "importance_delta")
    @classmethod
    def _clamp_unit(cls, v: float | None) -> float | None:
        if v is None:
            return v
        return max(0.0, min(1.0, v))


class MemoryReinforce(BaseModel):
    """Payload for reinforce operation."""

    importance_delta: float = Field(default_factory=lambda: _get_dynamics().reinforce_delta)
    valence_delta: float | None = Field(default=None)
    emotional_intensity_delta: float | None = Field(
        default=None,
        validation_alias=AliasChoices("arousal_delta", "emotional_intensity_delta"),
        serialization_alias="arousal_delta",
    )

    model_config = {
        "extra": "forbid",
        "validate_default": True,
    }

    @field_validator("importance_delta")
    @classmethod
    def _clamp_imp(cls, v: float) -> float:
        return max(-1.0, min(1.0, v))

    @field_validator("valence_delta")
    @classmethod
    def _clamp_val(cls, v: float | None) -> float | None:
        if v is None:
            return v
        return max(-1.0, min(1.0, v))

    @field_validator("emotional_intensity_delta")
    @classmethod
    def _clamp_int(cls, v: float | None) -> float | None:
        if v is None:
            return v
        return max(-1.0, min(1.0, v))


class MemoryRead(MemoryBase):
    """Full memory record as returned by the API."""

    id: str
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    valence: float = Field(0.0, ge=-1.0, le=1.0)
    emotional_intensity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("arousal", "emotional_intensity"),
        serialization_alias="arousal",
    )
    importance: float = Field(0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Query / search models
# ---------------------------------------------------------------------------
class MemoryQuery(BaseModel):
    """Vector or text query parameters."""

    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(10, ge=1, le=100)
    include_embeddings: bool = Field(False, description="Return raw vector embeddings in the response")
    metadata_filter: dict[str, Any] | None = Field(default=None, description="Optional metadata key/value filters")
    modality: str = Field("text", description="Modality of the query")
    language: str | None = Field(None, description="Language hint")

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.query or len(self.query) > 1_000:
            raise ValueError("query must be between 1 and 1000 characters")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")


class MemorySearchResult(MemoryRead):
    """Memory with an additional similarity score."""

    score: float = Field(..., ge=0.0, le=1.0)
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Health & monitoring models (used by health routes)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Global health report for the service."""

    status: str = Field(..., description="healthy | degraded | unhealthy | reindexing")
    timestamp: str
    uptime_seconds: int = Field(..., ge=0)
    version: str = Field(_SERVICE_VERSION, description="Service version")
    checks: dict[str, bool]
    memory_store_health: dict[str, Any]
    api_enabled: bool


class StatsResponse(BaseModel):
    """Aggregated runtime metrics for dashboards / automation."""

    total_memories: int = Field(..., ge=0)
    active_sessions: int = Field(..., ge=0)
    uptime_seconds: int = Field(..., ge=0)
    memory_store_stats: dict[str, Any]
    api_stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Generic success / error wrappers (optional helpers)
# ---------------------------------------------------------------------------
class SuccessResponse(BaseModel):
    message: str = "success"
    api_version: str = _API_VERSION


class ErrorResponse(BaseModel):
    detail: str
    api_version: str = _API_VERSION
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dynamics() -> SimpleNamespace:
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.config.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return dyn
    except Exception:  # pragma: no cover - optional settings
        return SimpleNamespace(initial_intensity=0.0, reinforce_delta=0.1, decay_rate=30.0)
