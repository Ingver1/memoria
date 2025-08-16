# schemas.py — Pydantic models for Unified Memory System API
#
# Version: v{__version__}
"""Centralised data‑contracts used by the REST API layer."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from memory_system.utils.pydantic_compat import import_pydantic

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
else:
    _pyd = import_pydantic()
    AliasChoices = _pyd.AliasChoices
    BaseModel = _pyd.BaseModel
    ConfigDict = _pyd.ConfigDict
    Field = _pyd.Field
    model_validator = _pyd.model_validator

from memory_system import __version__

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_API_VERSION: str = "v1"
_SERVICE_VERSION: str = __version__


# ---------------------------------------------------------------------------
# Authentication models
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    """Response model returned when issuing an access token."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")


# ---------------------------------------------------------------------------
# Memory domain models
# ---------------------------------------------------------------------------
class ExperienceCard(BaseModel):
    """Schema for structured experience cards."""

    situation: str | None = None
    approach: str | None = None
    tools: str | None = None
    result: str | None = None
    lesson: str | None = None
    antipattern: str | None = None
    verifiability: str | None = None
    source_hash: str | None = None
    lang: str | None = None
    summary: str | None = None
    summary_en: str | None = None
    success_count: int | None = Field(default=0, ge=0)
    trial_count: int | None = Field(default=0, ge=0)


class CodeVerifier(BaseModel):
    """Verification metadata for code memories."""

    type: Literal["code"] = "code"
    test_suite_path: str
    build_hash: str


class MathVerifier(BaseModel):
    """Verification metadata for mathematical memories."""

    type: Literal["math"] = "math"
    check: str


Verifier = Annotated[CodeVerifier | MathVerifier, Field(discriminator="type")]


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
    lang: str | None = Field(None, max_length=32, description="Detected language (BCP-47)")
    lang_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score for detected language"
    )
    memory_type: Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] = Field(
        "episodic"
    )
    pinned: bool = Field(False, description="Prevent memory from being forgotten")
    ttl_seconds: int | None = Field(None, ge=0)
    last_used: datetime | None = Field(None)
    success_score: float | None = Field(0.0)
    decay: float | None = Field(0.0)
    verifier: Verifier | None = Field(default=None, description="Optional verification details")

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
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata to attach to the memory",
    )
    schema_type: str | None = Field(
        default=None,
        description="Optional schema identifier for structured memories",
    )
    verifiability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in memory accuracy",
    )
    source_hash: str | None = Field(
        default=None,
        description="Hash of the original memory source",
    )
    provenance: list[str] | None = Field(
        default=None,
        description="List of provenance identifiers",
    )
    card: ExperienceCard | None = Field(
        default=None,
        description="Optional structured experience card",
    )


class MemoryBatchItem(BaseModel):
    """Payload item for batch memory creation."""

    text: str = Field(..., min_length=1, max_length=10_000)
    modality: str = Field("text", max_length=32)
    metadata: dict[str, Any] | None = Field(default=None)
    memory_type: Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] = Field(
        "episodic"
    )
    pinned: bool = Field(False)
    ttl_seconds: int | None = Field(None, ge=0)
    verifier: Verifier | None = Field(default=None)
    last_used: datetime | None = Field(None)
    success_score: float | None = Field(0.0)
    decay: float | None = Field(0.0)

    model_config = ConfigDict(extra="forbid")


class MemoryUpdate(BaseModel):
    """
    Payload for partial updates where all fields are optional.

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
    schema_type: str | None = Field(None)
    verifiability: float | None = Field(
        default=None,
        description="Set absolute verifiability",
    )
    source_hash: str | None = Field(default=None)
    provenance: list[str] | None = Field(default=None)
    memory_type: Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] | None = (
        Field(None)
    )
    pinned: bool | None = Field(None)
    ttl_seconds: int | None = Field(None, ge=0)
    last_used: datetime | None = Field(None)
    success_score: float | None = Field(None)
    decay: float | None = Field(None)
    verifier: Verifier | None = Field(default=None, description="Replace verification metadata")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _clamp_values(self) -> MemoryUpdate:
        for name in ("valence", "valence_delta"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, max(-1.0, min(1.0, v)))
        for name in (
            "emotional_intensity",
            "emotional_intensity_delta",
            "importance",
            "importance_delta",
            "verifiability",
        ):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, max(0.0, min(1.0, v)))
        return self


class MemoryReinforce(BaseModel):
    """Payload for reinforce operation."""

    importance_delta: float = Field(default_factory=lambda: _get_dynamics().reinforce_delta)
    valence_delta: float | None = Field(default=None)
    emotional_intensity_delta: float | None = Field(
        default=None,
        validation_alias=AliasChoices("arousal_delta", "emotional_intensity_delta"),
        serialization_alias="arousal_delta",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    @model_validator(mode="after")
    def _clamp_deltas(self) -> MemoryReinforce:
        self.importance_delta = max(-1.0, min(1.0, self.importance_delta))
        if self.valence_delta is not None:
            self.valence_delta = max(-1.0, min(1.0, self.valence_delta))
        if self.emotional_intensity_delta is not None:
            self.emotional_intensity_delta = max(-1.0, min(1.0, self.emotional_intensity_delta))
        return self


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
    score_parts: dict[str, float] | None = Field(
        default=None,
        description="Optional weighted score components",
    )


class MemoryAging(BaseModel):
    """Aging and retention metrics for a memory."""

    id: str
    age_days: float = Field(..., ge=0.0)
    retention: float = Field(..., ge=0.0, le=1.0)
    access_count: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Query / search models
# ---------------------------------------------------------------------------
class MemoryQuery(BaseModel):
    """Vector or text query parameters."""

    query: str = Field(..., min_length=1, max_length=1_000)
    top_k: int = Field(10, ge=1, le=100)
    include_embeddings: bool = Field(
        False, description="Return raw vector embeddings in the response"
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None, description="Optional metadata key/value filters"
    )
    modality: str = Field("text", description="Modality of the query")
    language: str | None = Field(None, description="Language hint")
    lang: str | None = Field(None, description="Detected language (BCP-47)")
    lang_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score for detected language"
    )
    context: dict[str, Any] | None = Field(
        default=None, description="Optional context for adaptive ranking"
    )
    channel: str = Field(
        "global",
        description="Memory channel to search. Only 'global' is supported",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.query or len(self.query) > 1_000:
            raise ValueError("query must be between 1 and 1000 characters")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if data.get("channel") not in (None, "global"):
            raise ValueError("Only 'global' channel is supported")


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
    ranking_min_score: float = Field(..., ge=0.0, description="Minimum accepted draft score")


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
    idempotency_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("Idempotency-Key", "idempotency_key"),
        serialization_alias="Idempotency-Key",
        description="Echoed idempotency key",
    )
    idempotent: bool | None = Field(
        default=None,
        description="Result of idempotency check",
    )


class ErrorResponse(BaseModel):
    detail: str
    api_version: str = _API_VERSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dynamics() -> SimpleNamespace:
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return cast("SimpleNamespace", dyn)
    except (ImportError, AttributeError):  # pragma: no cover - optional settings
        return SimpleNamespace(initial_intensity=0.0, reinforce_delta=0.1, decay_rate=30.0)
