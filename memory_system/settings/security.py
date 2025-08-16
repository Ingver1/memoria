from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from .base import BaseModel, ConfigDict, Field, PositiveInt, model_validator

try:  # optional dependency
    from cryptography.fernet import Fernet as FernetType, InvalidToken as InvalidTokenType

    _CRYPTO_OK = True
except Exception:  # pragma: no cover - cryptography not installed
    _CRYPTO_OK = False

    class InvalidTokenType(Exception):  # type: ignore[no-redef]
        ...

    class FernetType:  # type: ignore[no-redef]
        @staticmethod
        def generate_key() -> bytes:
            return b"0" * 32

        def __init__(self, key: bytes) -> None:
            self._k = key

        def encrypt(self, data: bytes) -> bytes:  # pragma: no cover - stub
            return data

        def decrypt(self, token: bytes) -> bytes:  # pragma: no cover - stub
            return token


Fernet = FernetType
InvalidToken = InvalidTokenType


from memory_system.utils.security import SecretStr, validate_fernet_key

if TYPE_CHECKING:  # pragma: no cover
    from memory_system.utils.security import KeyManager


class SecurityConfig(BaseModel):
    """Security related options."""

    encrypt_at_rest: bool = False
    encryption_key: SecretStr = SecretStr("")
    filter_pii: bool = True
    pii_log_details: bool = False
    privacy_mode: Literal["strict", "standard"] = "strict"
    telemetry_level: Literal["none", "aggregate"] = "aggregate"
    max_text_length: PositiveInt = 10_000
    max_stream_lines: PositiveInt = 10_000
    rate_limit_per_minute: PositiveInt = 1_000
    api_token: str = "your-secret-token-change-me"
    kms_backend: str | None = None
    kms_key_id: str | None = None
    kms_params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    def __init__(self, **data: Any) -> None:
        if data.get("encrypt_at_rest") and not _CRYPTO_OK:
            raise RuntimeError(
                "encrypt_at_rest requires the 'cryptography' package. "
                "Install optional dependency with `pip install ai-memory[security]`."
            )
        super().__init__(**data)
        if len(self.api_token) < 8:
            raise ValueError("API token must be at least 8 characters long")
        if self.encryption_key.get_secret_value():
            object.__setattr__(self, "encryption_key", self._validate_key(self.encryption_key))
        if self.encrypt_at_rest and not self.encryption_key.get_secret_value():
            key = os.getenv("TEST_FERNET_KEY")
            if key:
                object.__setattr__(self, "encryption_key", self._validate_key(key))
            else:
                try:
                    from memory_system.utils.security import CryptoContext

                    ctx = CryptoContext.from_env()
                    object.__setattr__(
                        self, "encryption_key", self._validate_key(ctx.get_active_key())
                    )
                except Exception:
                    key = Fernet.generate_key().decode()
                    object.__setattr__(self, "encryption_key", self._validate_key(key))

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name == "encrypt_at_rest" and value:
            if not _CRYPTO_OK:
                raise RuntimeError(
                    "encrypt_at_rest requires the 'cryptography' package. "
                    "Install optional dependency with `pip install ai-memory[security]`."
                )
            current = getattr(self, "encryption_key", SecretStr(""))
            if not current.get_secret_value():
                key = os.getenv("TEST_FERNET_KEY")
                if key:
                    object.__setattr__(self, "encryption_key", self._validate_key(key))
                else:
                    try:
                        from memory_system.utils.security import CryptoContext

                        ctx = CryptoContext.from_env()
                        object.__setattr__(
                            self,
                            "encryption_key",
                            self._validate_key(ctx.get_active_key()),
                        )
                    except Exception:
                        key = Fernet.generate_key().decode()
                        object.__setattr__(self, "encryption_key", self._validate_key(key))
        super().__setattr__(name, value)

    @staticmethod
    def _validate_key(value: str | SecretStr) -> SecretStr:
        secret = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not secret:
            return SecretStr("")
        try:
            validate_fernet_key(secret)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(
                "Invalid encryption key; expected URL-safe base64-encoded 32-byte key"
            ) from exc
        return SecretStr(secret)

    @model_validator(mode="after")
    def _validate_fields(self) -> SecurityConfig:
        if self.encryption_key.get_secret_value():
            object.__setattr__(self, "encryption_key", self._validate_key(self.encryption_key))
        if len(self.api_token) < 8:
            raise ValueError("API token must be at least 8 characters long")
        if self.kms_backend is not None:
            allowed = {"local", "aws", "vault"}
            if self.kms_backend not in allowed:
                raise ValueError(f"kms_backend must be one of {allowed}")
        backend = self.kms_backend
        params = self.kms_params
        if backend == "vault":
            required = {"url", "token", "secret_path"}
            missing = required - params.keys()
            if missing:
                raise ValueError(
                    f"kms_params missing required keys for vault backend: {sorted(missing)}"
                )
        elif backend == "aws":
            if not (self.kms_key_id or params.get("key_id") or params.get("KeyId")):
                raise ValueError("kms_key_id or kms_params['key_id'] required for aws backend")
        return self

    def build_key_manager(self) -> KeyManager:
        from memory_system.utils.security import (
            AWSKMSKeyManager,
            LocalKeyManager,
            VaultKeyManager,
        )

        backend = self.kms_backend or "local"
        if backend == "local":
            if not self.encryption_key.get_secret_value():
                raise ValueError("encryption_key must be provided for local kms backend")
            return LocalKeyManager(self.encryption_key.get_secret_value())
        if backend == "vault":
            return VaultKeyManager(**self.kms_params)
        if backend == "aws":
            params = dict(self.kms_params)
            if self.kms_key_id:
                params.setdefault("key_id", self.kms_key_id)
            return AWSKMSKeyManager(**params)
        raise ValueError(f"Unsupported kms_backend: {backend}")


__all__ = ["SecurityConfig"]
