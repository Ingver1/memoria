from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import quote

from memory_system import __version__
from memory_system.core.summarization import SummaryStrategy
from memory_system.utils.logging import RequestIdFilter, setup_request_id_logging

from .base import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    ValidationError,
    model_validator,
)
from .database import DatabaseConfig
from .performance import PerformanceConfig
from .security import SecurityConfig


class ModelConfig(BaseModel):
    """Embedding model and ANN index parameters."""

    model_name: str = "bge-m3"
    batch_add_size: PositiveInt = 128
    vector_dim: PositiveInt = 1_024
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    vector_dims: dict[str, PositiveInt] = Field(default_factory=lambda: {"text": 1_024})

    model_config = ConfigDict(frozen=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not data.get("vector_dims"):
            object.__setattr__(self, "vector_dims", {self.modalities[0]: self.vector_dim})
        else:
            first = self.modalities[0]
            object.__setattr__(self, "vector_dim", self.vector_dims[first])


class FaissConfig(BaseModel):
    """Fine-tuning parameters for FAISS indexes."""

    index_type: str = "HNSW"
    use_gpu: bool = Field(
        False,
        description=(
            "Use FAISS GPU acceleration when available. Requires the "
            "`faiss-gpu` package; if GPU support is missing the system will "
            "fall back to CPU mode and log a warning."
        ),
    )
    M: PositiveInt = 32
    ef_search: PositiveInt | None = None
    ef_construction: PositiveInt | None = None
    nlist: PositiveInt | None = None
    nprobe: PositiveInt = 8
    pq_m: PositiveInt | None = None
    pq_bits: PositiveInt | None = 8
    autotune: bool = False
    dataset_size: PositiveInt | None = None
    nlist_scale: float = 4.0
    ef_search_scale: float = 0.5
    ef_construction_scale: float = 2.0
    pq_m_div: PositiveInt = 24

    model_config = ConfigDict(frozen=True)


class CacheConfig(BaseModel):
    """Generic in-memory cache configuration."""

    size: PositiveInt = 1_000
    ttl_seconds: PositiveInt = 300

    model_config = ConfigDict(frozen=True)


class KnnLMConfig(BaseModel):
    """Configuration for kNN-LM interpolation."""

    lambda_: float = Field(0.5, alias="lambda", serialization_alias="lambda")
    cache_size: PositiveInt = 1_024

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _lambda_range(self) -> KnnLMConfig:
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda must be between 0 and 1")
        return self


class RAGConfig(BaseModel):
    """Configuration for RAG entropy-based routing."""

    entropy_threshold: float = 1.5

    model_config = ConfigDict(frozen=True)


class EffortBudget(BaseModel):
    """Budget limits for a single effort level."""

    max_k: PositiveInt = 10
    max_context_tokens: PositiveInt = 4_096
    max_cross_rerank_n: PositiveInt = 64
    timeout_seconds: float = 5.0

    model_config = ConfigDict(frozen=True)


class EffortConfig(BaseModel):
    """Budgets grouped by effort level."""

    low: EffortBudget = EffortBudget(
        max_k=4, max_context_tokens=2_048, max_cross_rerank_n=32, timeout_seconds=3.0
    )
    med: EffortBudget = EffortBudget()
    high: EffortBudget = EffortBudget(
        max_k=20, max_context_tokens=8_192, max_cross_rerank_n=128, timeout_seconds=10.0
    )

    model_config = ConfigDict(frozen=True)


class ReliabilityConfig(BaseModel):
    """Retries and backup settings."""

    max_retries: PositiveInt = 3
    retry_delay_seconds: float = 1.0
    backup_enabled: bool = True
    backup_interval_hours: PositiveInt = 24

    model_config = ConfigDict(frozen=True)


class SearchConfig(BaseModel):
    """Configuration for search specific behaviour."""

    w_lang: float = 0.0

    model_config = ConfigDict(frozen=True)


class RankingConfig(BaseModel):
    """Weighting for ranking memories via :func:`list_best`."""

    importance: float = 1.0
    emotional_intensity: float = 1.0
    valence_pos: float = 1.0
    valence_neg: float = 0.5
    tag_match: float = 1.0
    role_match: float = 1.0
    min_score: float = Field(
        0.0, ge=0.0, description="Minimum combined ranking score for search results"
    )
    adaptation: dict[str, dict[str, dict[str, float]]] = Field(
        default_factory=dict,
        description="Context-specific weight overrides",
    )
    use_cross_encoder: bool = False
    cross_encoder_provider: str = "sbert"
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    w_f: float = 0.0

    model_config = ConfigDict(frozen=True)


class CompositeWeightsConfig(BaseModel):
    """Weights for composite memory scoring in RAG routing."""

    alpha: float = 1.0
    beta: float = 2.0
    gamma: float = 0.15
    eta: float = 0.10
    delta: float = 0.5
    lambda_: float = Field(1.0, alias="lambda", serialization_alias="lambda")

    model_config = ConfigDict(frozen=True)


class FeedbackConfig(BaseModel):
    """Configuration for feedback sampling."""

    alpha: float = 1.0

    model_config = ConfigDict(frozen=True)


class WorkingMemoryConfig(BaseModel):
    """Configuration for in-process working memory."""

    budget: PositiveInt = 10

    model_config = ConfigDict(frozen=True)


class DecayWeights(BaseModel):
    """Weighting applied during memory decay calculations."""

    importance: float = 0.4
    emotional_intensity: float = 0.3
    valence: float = 0.3

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _clamp_unit(self) -> DecayWeights:
        object.__setattr__(self, "importance", max(0.0, min(1.0, self.importance)))
        object.__setattr__(
            self, "emotional_intensity", max(0.0, min(1.0, self.emotional_intensity))
        )
        object.__setattr__(self, "valence", max(0.0, min(1.0, self.valence)))
        return self


class DynamicsConfig(BaseModel):
    """Defaults for memory dynamics such as decay and reinforcement."""

    initial_intensity: float = 0.0
    reinforce_delta: float = 0.1
    decay_rate: float = 30.0
    decay_weights: DecayWeights = DecayWeights()
    decay_law: Literal["exponential", "logarithmic"] = "exponential"

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _clamp_values(self) -> DynamicsConfig:
        object.__setattr__(self, "initial_intensity", max(0.0, min(1.0, self.initial_intensity)))
        object.__setattr__(self, "reinforce_delta", max(0.0, min(1.0, self.reinforce_delta)))
        object.__setattr__(self, "decay_rate", max(0.0, self.decay_rate))
        return self


class APIConfig(BaseModel):
    """HTTP API options."""

    host: str = "0.0.0.0"
    port: NonNegativeInt = 8_000
    enable_cors: bool = True
    enable_api: bool = True
    enable_translation: bool = False
    translation_confidence_threshold: float = 0.5
    translation_cache_size: NonNegativeInt = 1_024
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = ConfigDict(frozen=True)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("APIConfig is immutable")
        super().__setattr__(name, value)

    @model_validator(mode="after")
    def _validate_port(self) -> APIConfig:
        if self.port != 0 and (self.port < 1_024 or self.port > 65_535):
            raise ValueError("port must be between 1024 and 65535")
        return self


class MonitoringConfig(BaseModel):
    """Metrics and diagnostics configuration."""

    enable_metrics: bool = True
    enable_rate_limiting: bool = True
    prom_port: PositiveInt = 9_100
    health_check_interval: PositiveInt = 30
    log_level: str = "INFO"

    model_config = ConfigDict(frozen=False)

    @model_validator(mode="after")
    def _validate_prom_port(self) -> MonitoringConfig:
        if self.prom_port < 1_024 or self.prom_port > 65_535:
            raise ValueError("prom_port must be between 1024 and 65535")
        return self


class MaintenanceConfig(BaseModel):
    """Background maintenance scheduling options."""

    summarize_interval_seconds: PositiveInt = 3_600
    summarize_threshold: float = 0.83
    forget_interval_seconds: PositiveInt = 3_600
    forget_min_total: PositiveInt = 1_000
    forget_retain_fraction: float = 0.85
    forget_ttl_seconds: PositiveInt | None = None
    forget_low_trust_threshold: float | None = None
    forget_high_threshold: float = 0.8
    forget_low_threshold: float = 0.2
    forget_window_days: PositiveInt = 14
    cluster_algorithm: Literal["greedy", "faiss", "auto"] = "auto"
    cluster_auto_threshold: PositiveInt = 1_000
    audit_interval_seconds: PositiveInt = 3_600
    audit_drift_threshold: PositiveInt = 100

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _fraction_range(self) -> MaintenanceConfig:
        for name in (
            "summarize_threshold",
            "forget_retain_fraction",
            "forget_low_trust_threshold",
            "forget_high_threshold",
            "forget_low_threshold",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError("value must be between 0 and 1")
        return self


class UnifiedSettings(BaseModel):
    """Aggregate all configuration sections."""

    version: str = __version__
    profile: str = "development"
    summary_strategy: str | SummaryStrategy = "head2tail"
    database: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()
    faiss: FaissConfig = FaissConfig()
    security: SecurityConfig = SecurityConfig()
    cache: CacheConfig = CacheConfig()
    token_cache: CacheConfig = CacheConfig()
    performance: PerformanceConfig = PerformanceConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    search: SearchConfig = SearchConfig()
    ranking: RankingConfig = RankingConfig()
    composite: CompositeWeightsConfig = CompositeWeightsConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    working_memory: WorkingMemoryConfig = WorkingMemoryConfig()
    knn_lm: KnnLMConfig = KnnLMConfig()
    rag: RAGConfig = RAGConfig()
    effort: EffortConfig = EffortConfig()
    dynamics: DynamicsConfig = DynamicsConfig()
    api: APIConfig = APIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    maintenance: MaintenanceConfig = MaintenanceConfig()

    model_config = ConfigDict()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        def _clone(cfg_cls: type[BaseModel], value: Any) -> BaseModel:
            data_dict = value.model_dump() if isinstance(value, cfg_cls) else value
            return cfg_cls(**data_dict)

        object.__setattr__(self, "database", _clone(DatabaseConfig, self.database))
        object.__setattr__(self, "model", _clone(ModelConfig, self.model))
        object.__setattr__(self, "faiss", _clone(FaissConfig, self.faiss))
        object.__setattr__(self, "security", _clone(SecurityConfig, self.security))
        object.__setattr__(self, "cache", _clone(CacheConfig, self.cache))
        object.__setattr__(self, "token_cache", _clone(CacheConfig, self.token_cache))
        object.__setattr__(self, "performance", _clone(PerformanceConfig, self.performance))
        object.__setattr__(self, "reliability", _clone(ReliabilityConfig, self.reliability))
        object.__setattr__(self, "search", _clone(SearchConfig, self.search))
        object.__setattr__(self, "ranking", _clone(RankingConfig, self.ranking))
        object.__setattr__(self, "composite", _clone(CompositeWeightsConfig, self.composite))
        object.__setattr__(self, "feedback", _clone(FeedbackConfig, self.feedback))
        object.__setattr__(self, "working_memory", _clone(WorkingMemoryConfig, self.working_memory))
        object.__setattr__(self, "knn_lm", _clone(KnnLMConfig, self.knn_lm))
        object.__setattr__(self, "rag", _clone(RAGConfig, self.rag))
        object.__setattr__(self, "effort", _clone(EffortConfig, self.effort))
        object.__setattr__(self, "dynamics", _clone(DynamicsConfig, self.dynamics))
        object.__setattr__(self, "api", _clone(APIConfig, self.api))
        object.__setattr__(self, "monitoring", _clone(MonitoringConfig, self.monitoring))

        envs: dict[str, str] = {}
        env_file_val = self.model_config.get("env_file")
        env_path = Path(str(env_file_val)) if env_file_val else None
        if env_path and env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    envs.setdefault(k.strip(), v.strip())
        envs.update(os.environ)

        def coerce(val: str, typ: Any) -> Any:
            if typ is bool:
                return val.lower() in {"1", "true", "yes", "on"}
            if typ is int:
                return int(val)
            if typ is Path:
                return Path(val)
            return val

        prefix = str(self.model_config.get("env_prefix", ""))
        aliases: dict[str, tuple[str, str]] = {"privacy_mode": ("security", "privacy_mode")}
        for key, value in envs.items():
            trimmed = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if "__" not in trimmed:
                field = trimmed.lower()
                if hasattr(self, field):
                    current = getattr(self, field)
                    object.__setattr__(self, field, coerce(value, type(current)))
                elif field in aliases:
                    section, field_name = aliases[field]
                    cfg = getattr(self, section)
                    current = getattr(cfg, field_name)
                    object.__setattr__(cfg, field_name, coerce(value, type(current)))
                continue
            section, field = trimmed.split("__", 1)
            section = section.lower()
            field = field.lower()
            if hasattr(self, section):
                cfg = getattr(self, section)
                if hasattr(cfg, field):
                    current = getattr(cfg, field)
                    object.__setattr__(cfg, field, coerce(value, type(current)))
        paths = [
            self.database.db_path,
            self.database.vec_path,
            self.database.cache_path,
        ]
        fixed: list[Path] = []
        for p in paths:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fixed.append(p)
        object.__setattr__(self.database, "db_path", fixed[0])
        object.__setattr__(self.database, "vec_path", fixed[1])
        object.__setattr__(self.database, "cache_path", fixed[2])
        object.__setattr__(self, "storage", SimpleNamespace(database_url=self.get_database_url()))

    @classmethod
    def for_testing(cls) -> UnifiedSettings:
        test_model = ModelConfig(vector_dim=3, vector_dims={"text": 3})
        return cls(
            profile="testing",
            database=DatabaseConfig(wal_checkpoint_interval=0.1, wal_checkpoint_writes=5),
            cache=CacheConfig(size=100, ttl_seconds=10),
            token_cache=CacheConfig(size=100, ttl_seconds=10),
            performance=PerformanceConfig(max_workers=2, cache_size=100, cache_ttl_seconds=10),
            monitoring=MonitoringConfig(enable_metrics=False, health_check_interval=5),
            api=APIConfig(port=0),
            security=SecurityConfig(api_token="test-token-12345678", rate_limit_per_minute=10_000),
            model=test_model,
        )

    @classmethod
    def for_production(cls) -> UnifiedSettings:
        return cls(
            profile="production",
            database=DatabaseConfig(connection_pool_size=20),
            cache=CacheConfig(size=5_000),
            token_cache=CacheConfig(size=5_000),
            performance=PerformanceConfig(max_workers=8, cache_size=5_000),
            security=SecurityConfig(encrypt_at_rest=True, filter_pii=True),
        )

    @classmethod
    def for_development(cls) -> UnifiedSettings:
        return cls(
            profile="development",
            database=DatabaseConfig(connection_pool_size=5),
            cache=CacheConfig(size=500),
            token_cache=CacheConfig(size=500),
            performance=PerformanceConfig(max_workers=2, cache_size=500),
            monitoring=MonitoringConfig(
                enable_metrics=True, log_level="DEBUG", health_check_interval=10
            ),
            api=APIConfig(enable_cors=True),
        )

    def get_database_url(self) -> str:
        if self.database.url:
            return self.database.url
        scheme = "sqlite+sqlcipher" if self.security.encrypt_at_rest else "sqlite"
        url = f"{scheme}:///{self.database.db_path}"
        if self.security.encrypt_at_rest:
            url = f"{url}?cipher_secret={quote(self.security.encryption_key.get_secret_value())}"
        return url

    def validate_production_ready(self) -> list[str]:
        issues: list[str] = []

        if not self.security.api_token or self.security.api_token == "your-secret-token-change-me":
            issues.append("API token is not set")

        if self.profile != "production":
            issues.append("Profile is not production")

        if not self.security.encrypt_at_rest:
            issues.append("Encryption at rest disabled")

        if not self.security.filter_pii:
            issues.append("PII filtering disabled")

        if not self.monitoring.enable_metrics or self.security.telemetry_level != "aggregate":
            issues.append("Metrics disabled")

        return issues

    def get_config_summary(self) -> dict[str, Any]:
        def scrub(obj: BaseModel) -> dict[str, Any]:
            data: dict[str, Any] = obj.model_dump()
            data.pop("api_token", None)
            data.pop("encryption_key", None)
            if isinstance(obj, SecurityConfig):
                data["has_key"] = bool(self.security.encryption_key.get_secret_value())
            return data

        return {
            "version": self.version,
            "profile": self.profile,
            "database": scrub(self.database),
            "model": scrub(self.model),
            "faiss": scrub(self.faiss),
            "security": scrub(self.security),
            "performance": scrub(self.performance),
            "reliability": scrub(self.reliability),
            "ranking": scrub(self.ranking),
            "dynamics": scrub(self.dynamics),
            "api": scrub(self.api),
            "monitoring": scrub(self.monitoring),
            "maintenance": scrub(self.maintenance),
        }

    def save_to_file(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            data = self.model_dump(mode="json")
            data.pop("storage", None)
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> UnifiedSettings:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            data.pop("storage", None)
        return cls(**data)


def configure_logging(settings: UnifiedSettings | None = None) -> None:
    settings = settings or UnifiedSettings()
    level = getattr(logging, settings.monitoring.log_level.upper(), logging.INFO)
    setup_request_id_logging()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s",
    )
    logging.getLogger().addFilter(RequestIdFilter())
    if settings.security.filter_pii:
        from memory_system.utils.security import PIILoggingFilter

        logging.getLogger().addFilter(PIILoggingFilter())


def get_settings(env: str | None = None) -> UnifiedSettings:
    env = env or os.getenv("AI_ENV", "development")
    if env == "production":
        return UnifiedSettings.for_production()
    if env == "testing":
        return UnifiedSettings.for_testing()
    if env == "development":
        return UnifiedSettings.for_development()
    return UnifiedSettings()


__all__ = [
    "APIConfig",
    "CacheConfig",
    "CompositeWeightsConfig",
    "DecayWeights",
    "DynamicsConfig",
    "EffortBudget",
    "EffortConfig",
    "FaissConfig",
    "FeedbackConfig",
    "KnnLMConfig",
    "MaintenanceConfig",
    "ModelConfig",
    "MonitoringConfig",
    "RAGConfig",
    "RankingConfig",
    "ReliabilityConfig",
    "UnifiedSettings",
    "configure_logging",
    "get_settings",
]
