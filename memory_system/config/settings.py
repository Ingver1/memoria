"""Application configuration models for the Unified Memory System."""

from __future__ import annotations

import json
import logging.config
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from memory_system import __version__
from pydantic import BaseModel, Field, PositiveInt, ValidationError, field_validator
from pydantic_settings import BaseSettings

__all__ = [
    "DatabaseConfig",
    "ModelConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "ReliabilityConfig",
    "APIConfig",
    "MonitoringConfig",
    "UnifiedSettings",
    "configure_logging",
    "get_settings",
]


class DatabaseConfig(BaseModel):
    """Paths and connection limits for the storage backend."""

    db_path: Path = Path("data/memory.db")
    vec_path: Path = Path("data/memory.vectors")
    cache_path: Path = Path("data/memory.cache")
    connection_pool_size: PositiveInt = 10

    model_config = {"frozen": True}

    @field_validator("db_path", "vec_path", "cache_path")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        return v if isinstance(v, Path) else Path(v)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - simple immutability check
        if self.model_config.get("frozen") and name in self.__dict__:
            raise ValidationError("DatabaseConfig is immutable")
        super().__setattr__(name, value)


class ModelConfig(BaseModel):
    """Embedding model and ANN index parameters."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_add_size: PositiveInt = 128
    hnsw_m: PositiveInt = 32
    hnsw_ef_construction: PositiveInt = 200
    hnsw_ef_search: PositiveInt = 100
    vector_dim: PositiveInt = 384

    model_config = {"frozen": True}


class SecurityConfig(BaseModel):
    """Security related options."""

    encrypt_at_rest: bool = False
    encryption_key: str = ""
    filter_pii: bool = True
    max_text_length: PositiveInt = 10_000
    rate_limit_per_minute: PositiveInt = 1_000
    api_token: str = "your-secret-token-change-me"

    model_config = {"frozen": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if len(self.api_token) < 8:
            raise ValidationError("API token must be at least 8 characters long")
        if self.encryption_key:
            self._validate_key(self.encryption_key)
        if self.encrypt_at_rest and not self.encryption_key:
            object.__setattr__(self, "encryption_key", Fernet.generate_key().decode())

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name == "encrypt_at_rest" and value and not getattr(self, "encryption_key", ""):
            object.__setattr__(self, "encryption_key", Fernet.generate_key().decode())
        super().__setattr__(name, value)

    @field_validator("encryption_key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        if not value:
            return value
        try:
            Fernet(value.encode())
        except Exception as exc:  # pragma: no cover
            raise ValidationError("Invalid encryption key") from exc
        return value

    @field_validator("api_token")
    @classmethod
    def _validate_token(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("API token must be at least 8 characters long")
        return value


class PerformanceConfig(BaseModel):
    """Tuning knobs for throughput and caching."""

    max_workers: PositiveInt = 4
    cache_size: PositiveInt = 1_000
    cache_ttl_seconds: PositiveInt = 300
    rebuild_interval_seconds: PositiveInt = 3600

    model_config = {"frozen": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._workers_range(self.max_workers)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("PerformanceConfig is immutable")
        super().__setattr__(name, value)

    @field_validator("max_workers")
    @classmethod
    def _workers_range(cls, value: int) -> int:
        if value < 1 or value > 32:
            raise ValidationError("max_workers must be between 1 and 32")
        return value


class ReliabilityConfig(BaseModel):
    """Retries and backup settings."""

    max_retries: PositiveInt = 3
    retry_delay_seconds: float = 1.0
    backup_enabled: bool = True
    backup_interval_hours: PositiveInt = 24

    model_config = {"frozen": True}


class APIConfig(BaseModel):
    """HTTP API options."""

    host: str = "0.0.0.0"
    port: PositiveInt = 8000
    enable_cors: bool = True
    enable_api: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = {"frozen": True}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_port(self.port)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name in self.__dict__ and self.model_config.get("frozen"):
            raise ValidationError("APIConfig is immutable")
        super().__setattr__(name, value)

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value < 0 or value > 65_535 or (value != 0 and value < 1024):
            raise ValidationError("port must be between 1024 and 65535")
        return value


class MonitoringConfig(BaseModel):
    """Metrics and diagnostics configuration."""

    enable_metrics: bool = True
    enable_rate_limiting: bool = True
    prom_port: PositiveInt = 9_100
    health_check_interval: PositiveInt = 30
    log_level: str = "INFO"

    model_config = {"frozen": False}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_prom_port(self.prom_port)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        super().__setattr__(name, value)

    @field_validator("prom_port")
    @classmethod
    def _validate_prom_port(cls, value: int) -> int:
        if value < 1024 or value > 65_535:
            raise ValidationError("prom_port must be between 1024 and 65535")
        return value


class UnifiedSettings(BaseSettings):
    """Aggregate all configuration sections."""

    version: str = __version__
    profile: str = "development"
    database: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()
    security: SecurityConfig = SecurityConfig()
    performance: PerformanceConfig = PerformanceConfig()
    reliability: ReliabilityConfig = ReliabilityConfig()
    api: APIConfig = APIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    model_config = {"env_prefix": "AI_", "env_file": ".env"}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # environment variable overrides
        envs: dict[str, str] = {}
        env_file = self.model_config.get("env_file")
        if env_file and Path(env_file).exists():
            for line in Path(env_file).read_text().splitlines():
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
        for key, value in envs.items():
            if "__" not in key:
                continue
            trimmed = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            section, field = trimmed.split("__", 1)
            section = section.lower()
            field = field.lower()
            if hasattr(self, section):
                cfg = getattr(self, section)
                if hasattr(cfg, field):
                    current = getattr(cfg, field)
                    object.__setattr__(cfg, field, coerce(value, type(current)))
        # Ensure Path objects and existing directories
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
    def for_testing(cls) -> "UnifiedSettings":
        return cls(
            profile="testing",
            database=DatabaseConfig(model_config={"frozen": False}),
            performance=PerformanceConfig(max_workers=2, cache_size=100, cache_ttl_seconds=10),
            monitoring=MonitoringConfig(enable_metrics=False, health_check_interval=5),
            api=APIConfig(port=0),
            security=SecurityConfig(api_token="test-token-12345678", rate_limit_per_minute=10_000),
        )

    @classmethod
    def for_production(cls) -> "UnifiedSettings":
        return cls(
            profile="production",
            database=DatabaseConfig(connection_pool_size=20),
            performance=PerformanceConfig(max_workers=8, cache_size=5_000),
            security=SecurityConfig(encrypt_at_rest=True, filter_pii=True),
        )

    @classmethod
    def for_development(cls) -> "UnifiedSettings":
        return cls(
            profile="development",
            database=DatabaseConfig(connection_pool_size=5),
            performance=PerformanceConfig(max_workers=2, cache_size=500),
            monitoring=MonitoringConfig(enable_metrics=True, log_level="DEBUG", health_check_interval=10),
            api=APIConfig(enable_cors=True),
        )

    def get_database_url(self) -> str:
        return f"sqlite:///{self.database.db_path}"

    def validate_production_ready(self) -> list[str]:
        """Return any configuration issues that make the app unsafe for production.

        This checks the API token, runtime profile, encryption settings,
        whether PII filtering is enabled, and if metrics collection is active.
        """

        issues: list[str] = []

        if not self.security.api_token or self.security.api_token == "your-secret-token-change-me":
            issues.append("API token is not set")

        if self.profile != "production":
            issues.append("Profile is not production")

        if not self.security.encrypt_at_rest:
            issues.append("Encryption at rest disabled")

        if not self.security.filter_pii:
            issues.append("PII filtering disabled")

        if not self.monitoring.enable_metrics:
            issues.append("Metrics disabled")

        return issues

    def get_config_summary(self) -> dict[str, Any]:
        def scrub(obj: BaseModel) -> dict[str, Any]:
            data = obj.model_dump()
            data.pop("api_token", None)
            data.pop("encryption_key", None)
            if isinstance(obj, SecurityConfig):
                data["has_key"] = bool(self.security.encryption_key)
            return data

        return {
            "version": self.version,
            "profile": self.profile,
            "database": scrub(self.database),
            "model": scrub(self.model),
            "security": scrub(self.security),
            "performance": scrub(self.performance),
            "reliability": scrub(self.reliability),
            "api": scrub(self.api),
            "monitoring": scrub(self.monitoring),
        }

    def save_to_file(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(exclude={"storage"}), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "UnifiedSettings":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            data.pop("storage", None)
        return cls(**data)


def configure_logging(settings: UnifiedSettings | None = None) -> None:
    """Configure basic console logging for the application."""

    settings = settings or UnifiedSettings()

    level = getattr(logging, settings.monitoring.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_settings(env: str | None = None) -> UnifiedSettings:
    env = env or os.getenv("AI_ENV", "development")
    if env == "production":
        return UnifiedSettings.for_production()
    if env == "testing":
        return UnifiedSettings.for_testing()
    if env == "development":
        return UnifiedSettings.for_development()
    return UnifiedSettings()
