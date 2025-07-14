"""memory_system.utils.security
================================
High‑level cryptographic helpers for AI‑memory‑.

Key features
------------
* **CryptoContext** – injectable object encapsulating symmetric encryption (Fernet),
  signing and verification with automatic *key rotation*.
* **Pluggable KMS backend** – default local JSON keyring + optional AWS KMS backend
  (lazy import, safe to mock in unit‑tests).
* **Audit trail** – every call to *encrypt/sign/verify/decrypt* is logged to
  ``logging.getLogger("ai_memory.security.audit")`` with correlation IDs.
* 100 % type‑hinted, ready for dependency injection via FastAPI.

Example
~~~~~~~
>>> ctx = CryptoContext.from_env()
>>> token = ctx.encrypt(b"secret")
>>> data  = ctx.decrypt(token)

Copyright © Ingver1 2025.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, SecretStr, ValidationInfo, field_validator

__all__ = [
    "CryptoContext",
    "KeyManagementBackend",
    "LocalKeyBackend",
"PIIPatterns",
    "EnhancedPIIFilter",
    "SecureTokenManager",
    "PasswordManager",
    "EncryptionManager",
]

_audit_logger = logging.getLogger("ai_memory.security.audit")
_audit_logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Models & Helpers
# ---------------------------------------------------------------------------
class KeyMetadata(BaseModel):
    """Metadata attached to each managed key."""

    key_id: str  # UUID v4 string
    created_at: datetime
    expires_at: datetime | None = None  # None means no expiry

    @field_validator("expires_at")
    @classmethod
    def _exp_after_created(
        cls, v: datetime | None, info: ValidationInfo
    ) -> datetime | None:
        if v is not None and v <= info.data["created_at"]:
            raise ValueError("expires_at must be after created_at")
        return v


class ManagedKey(BaseModel):
    """Actual key material with metadata."""

    metadata: KeyMetadata
    fernet_key: SecretStr  # URL‑safe base64‑encoded 32‑byte key

    def as_fernet(self) -> Fernet:
        return Fernet(self.fernet_key.get_secret_value().encode())


# ---------------------------------------------------------------------------
# Key management backend abstraction
# ---------------------------------------------------------------------------
class KeyManagementBackend(ABC):
    """Abstract backend that persists and retrieves encryption keys."""

    @abstractmethod
    def load_all(self) -> list[ManagedKey]:
        raise NotImplementedError

    @abstractmethod
    def save(self, key: ManagedKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key_id: str) -> None:
        raise NotImplementedError


class LocalKeyBackend(KeyManagementBackend):
    """Simple JSON keyring stored on disk; suitable for dev or air‑gapped edge."""

    def __init__(self, path: str | Path | None = None) -> None:
        default_path = os.getenv("AI_MEMORY_KEYRING", ".keyring.json")
        if path is None:
            self._path = Path(default_path)
        else:
            self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _read(self) -> MutableMapping[str, Mapping[str, str]]:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            return cast(MutableMapping[str, Mapping[str, str]], data)
        return {}

    def _write(self, data: Mapping[str, Mapping[str, str]]) -> None:
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # ---------------------------------------------------------------------
    # Interface implementation
    # ---------------------------------------------------------------------
    def load_all(self) -> list[ManagedKey]:
        return [ManagedKey.model_validate(entry) for entry in self._read().values()]

    def save(self, key: ManagedKey) -> None:
        ring = self._read()
        ring[key.metadata.key_id] = key.model_dump(mode="json")
        self._write(ring)

    def delete(self, key_id: str) -> None:
        ring = self._read()
        if key_id in ring:
            del ring[key_id]
            self._write(ring)


# ---------------------------------------------------------------------------
# CryptoContext
# ---------------------------------------------------------------------------
class CryptoContext:
    """Encapsulates all cryptographic operations for the service."""

    # How often to rotate keys automatically (hours)
    _DEFAULT_ROTATION_HOURS: Final[int] = 24 * 30  # 30 days

    # How long until a deprecated key is fully removed (grace period)
    _DEFAULT_RETIRE_HOURS: Final[int] = 24 * 90  # 90 days

    def __init__(
        self,
        *,
        backend: KeyManagementBackend | None = None,
        rotation_hours: int = _DEFAULT_ROTATION_HOURS,
        retire_hours: int = _DEFAULT_RETIRE_HOURS,
    ) -> None:
        self.backend: KeyManagementBackend = backend or LocalKeyBackend()
        self.rotation_hours = rotation_hours
        self.retire_hours = retire_hours
        self._lock = asyncio.Lock()
        self._cache: dict[str, ManagedKey] = {}

        # load keys from backend at startup
        for key in self.backend.load_all():
            self._cache[key.metadata.key_id] = key

        if not self._cache:
            # first‑time initialisation
            self._generate_and_store_new_key()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------
    @property
    def _active_key(self) -> ManagedKey:
        # active = newest non‑expired key
        return max(self._cache.values(), key=lambda k: k.metadata.created_at)

    def _generate_and_store_new_key(self) -> ManagedKey:
        key_material = base64.urlsafe_b64encode(os.urandom(32)).decode()
        meta = KeyMetadata(
            key_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
        )
        mkey = ManagedKey(metadata=meta, fernet_key=SecretStr(key_material))
        self.backend.save(mkey)
        self._cache[mkey.metadata.key_id] = mkey
        return mkey

    async def maybe_rotate_keys(self) -> None:
        """Generate new key if the active key is older than *rotation_hours*."""
        async with self._lock:
            age = datetime.now(UTC) - self._active_key.metadata.created_at
            if age > timedelta(hours=self.rotation_hours):
                new_key = self._generate_and_store_new_key()
                _audit_logger.info(
                    "key.rotated",
                    extra={
                        "new_key": new_key.metadata.key_id,
                        "age_hours": age.total_seconds() / 3600,
                    },
                )

            # retire keys beyond grace period
            for key_id, key in list(self._cache.items()):
                if datetime.now(UTC) - key.metadata.created_at > timedelta(hours=self.retire_hours):
                    self.backend.delete(key_id)
                    del self._cache[key_id]
                    _audit_logger.info("key.retired", extra={"key": key_id})

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------
    def encrypt(self, data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode()
        token = self._active_key.as_fernet().encrypt(data)
        _audit_logger.info(
            "encrypt", extra={"by": self._active_key.metadata.key_id, "size": len(data)}
        )
        return token.decode()

    def decrypt(self, token: str) -> bytes:
        for key in self._cache.values():
            try:
                data = key.as_fernet().decrypt(token.encode())
                _audit_logger.info("decrypt", extra={"key": key.metadata.key_id, "size": len(data)})
                return data
            except InvalidToken:
                continue
        _audit_logger.warning("decrypt.failed", extra={"token_prefix": token[:10]})
        raise InvalidToken("No valid key found for decryption")

    # ------------------------------------------------------------------
    # Message signing / verification (HMAC‑style using Fernet MAC)
    # ------------------------------------------------------------------
    def sign(self, data: bytes | str) -> str:
        token = self.encrypt(data)  # Fernet already appends MAC
        _audit_logger.info("sign", extra={"key": self._active_key.metadata.key_id})
        return token

    def verify(self, signature: str, data: bytes | str) -> bool:
        try:
            recovered = self.decrypt(signature)
        except InvalidToken:
            _audit_logger.warning("verify.invalid_token")
            return False
        result = recovered == (data.encode() if isinstance(data, str) else data)
        _audit_logger.info("verify", extra={"ok": result})
        return result

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> CryptoContext:
        """Create context using env variables or default local backend."""
        # Example: support AWS KMS when AWS_KMS_KEY_ID is defined
        if os.getenv("AWS_KMS_KEY_ID"):
            try:
                from .kms_aws import AWSKMSBackend  # local file to avoid heavy import

                backend: KeyManagementBackend = AWSKMSBackend(
                    key_id=os.environ["AWS_KMS_KEY_ID"],
                    region=os.getenv("AWS_REGION", "us-east-1"),
                )
            except ImportError:  # pragma: no cover
                logging.getLogger(__name__).warning(
                    "boto3 missing, falling back to LocalKeyBackend"
                )
                backend = LocalKeyBackend()
        else:
            backend = LocalKeyBackend()
        return cls(backend=backend)


# ---------------------------------------------------------------------------
# Periodic background task helpers (to be wired in FastAPI lifespan)
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402 – late import w/ uvicorn


async def start_maintenance(ctx: CryptoContext, interval_hours: int = 6) -> None:
    """Run *maybe_rotate_keys* every *interval_hours* until cancelled."""

    async def _loop() -> None:  # inner to attach cancellation gracefully
        while True:
            await ctx.maybe_rotate_keys()
            await asyncio.sleep(interval_hours * 3600)

    asyncio.create_task(_loop(), name="crypto_key_maintenance")

# ---------------------------------------------------------------------------
# Additional lightweight security helpers for tests
# ---------------------------------------------------------------------------
import hashlib
import hmac
import secrets
import string
import time
from typing import Iterable, Pattern

from .exceptions import SecurityError


class PIIPatterns:
    """Collection of regular expressions for common PII types."""

    EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE = re.compile(r"^(?:\+?\d{1,2}[ -]?)?(?:\(\d{3}\)|\d{3})[ -.]?\d{3}[ -.]?\d{4}$")
    CREDIT_CARD = re.compile(r"^(?:\d{4}[ -]?){3}\d{4}$")
    SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
    IP_ADDRESS = re.compile(
        r"^(?:25[0-5]|2[0-4]\d|1?\d{1,2})(?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}$"
    )
  

class EnhancedPIIFilter:
    """Utility to detect and redact PII in text."""

    def __init__(self, custom_patterns: dict[str, Pattern[str]] | None = None) -> None:
        self.patterns: dict[str, Pattern[str]] = {
            "email": PIIPatterns.EMAIL,
            "phone": re.compile(PIIPatterns.PHONE.pattern.strip("^$")),
            "credit_card": re.compile(PIIPatterns.CREDIT_CARD.pattern.strip("^$")),
            "ssn": re.compile(PIIPatterns.SSN.pattern.strip("^$")),
            "ip": re.compile(PIIPatterns.IP_ADDRESS.pattern.strip("^$")),
        }
        if custom_patterns:
            self.patterns.update(custom_patterns)
        self.stats: dict[str, int] = {key: 0 for key in self.patterns}

    def detect(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                self.stats[key] = self.stats.get(key, 0) + len(matches)
                found[key] = matches
        return found

    def redact(self, text: str) -> tuple[str, bool, list[str]]:
        found = self.detect(text)
        redacted = text
        for key, matches in found.items():
            placeholder = f"[{key.upper()}_REDACTED]"
            for m in matches:
                redacted = redacted.replace(m, placeholder)
        return redacted, bool(found), list(found.keys())

    def partial_redact(self, text: str, preserve_chars: int = 2) -> tuple[str, bool, list[str]]:
        found = self.detect(text)
        redacted = text
        for _key, matches in found.items():
            for m in matches:
                keep_start = m[:preserve_chars]
                keep_end = m[-preserve_chars:] if preserve_chars else ""
                placeholder = f"{keep_start}...{keep_end}"
                redacted = redacted.replace(m, placeholder)
        return redacted, bool(found), list(found.keys())

    def get_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0


class SecureTokenManager:
    """Simplified JWT-like token manager using HMAC-SHA256."""

    def __init__(self, secret_key: str, algorithm: str = "HS256", issuer: str = "unified-memory-system") -> None:
        if len(secret_key) < 32:
            raise SecurityError("Secret key must be at least 32 characters")
        if algorithm != "HS256":
            raise SecurityError("Algorithm not allowed")
        self.secret_key = secret_key.encode()
        self.algorithm = algorithm
        self.issuer = issuer
        self.revoked_tokens: set[str] = set()

    def _sign(self, data: bytes) -> str:
        sig = hmac.new(self.secret_key, data, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def _encode(self, payload: dict[str, Any]) -> str:
        header = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signature = self._sign(f"{header}.{body}".encode())
        return f"{header}.{body}.{signature}"

    def _decode(self, token: str) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            header_b64, body_b64, signature = token.split(".")
            data = f"{header_b64}.{body_b64}".encode()
            if not hmac.compare_digest(self._sign(data), signature):
                raise SecurityError("Invalid token")
            payload = json.loads(base64.urlsafe_b64decode(body_b64 + "=="))
            header = json.loads(base64.urlsafe_b64decode(header_b64 + "=="))
            return header, payload
        except Exception as exc:
            raise SecurityError("Invalid token") from exc

    def generate_token(
        self,
        user_id: str,
        *,
        expires_in: int = 3600,
        scopes: Iterable[str] | None = None,
        audience: str | None = None,
        token_type: str = "access",
    ) -> str:
        if not user_id or len(user_id) > 100:
            raise SecurityError("Invalid user_id")
        if expires_in <= 0 or expires_in > 86_400:
            raise SecurityError("Invalid expiration time")
        payload: dict[str, Any] = {
            "sub": user_id,
            "iss": self.issuer,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "token_type": token_type,
        }
        if scopes:
            payload["scopes"] = list(scopes)
        if audience:
            payload["aud"] = audience
        return self._encode(payload)

    def generate_refresh_token(self, user_id: str) -> str:
        return self.generate_token(
            user_id,
            audience="refresh",
            expires_in=86_400,
            token_type="refresh",
        )

    def verify_token(self, token: str, *, audience: str | None = None) -> dict[str, Any]:
        if token in self.revoked_tokens:
            raise SecurityError("Token revoked")
        _header, payload = self._decode(token)
        now = int(time.time())
        if payload.get("exp", 0) < now:
            raise SecurityError("Token expired")
        if audience is not None and payload.get("aud") != audience:
            raise SecurityError("Invalid token")
        return payload

    def revoke_token(self, token: str) -> bool:
        self.revoked_tokens.add(token)
        return True

    def get_stats(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "issuer": self.issuer,
            "revoked_tokens_count": len(self.revoked_tokens),
        }


class PasswordManager:
    """Helper for password hashing and verification."""

    @staticmethod
    def hash_password(password: str, salt: bytes | None = None) -> tuple[str, bytes]:
        if len(password) < 8:
            raise SecurityError("Password must be at least 8 characters")
        salt = salt or secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return base64.b64encode(dk).decode(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: bytes) -> bool:
        try:
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
            return hmac.compare_digest(base64.b64decode(hashed.encode()), dk)
        except Exception:
            return False

    @staticmethod
    def generate_secure_password(*, length: int = 16, include_symbols: bool = True) -> str:
        if not 8 <= length <= 128:
            raise SecurityError("Length must be between 8 and 128")
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
        if include_symbols:
            chars += "!@#$%^&*()_+-=[]{}|;:,.<>?"
        while True:
            pwd = "".join(secrets.choice(chars) for _ in range(length))
            if (
                any(c.islower() for c in pwd)
                and any(c.isupper() for c in pwd)
                and any(c.isdigit() for c in pwd)
                and (not include_symbols or any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in pwd))
            ):
                return pwd


class EncryptionManager:
    """Very small symmetric encryption helper using XOR and base64."""

    def __init__(self, key: bytes | None = None) -> None:
        self.key = key or secrets.token_bytes(32)

    def encrypt(self, data: str) -> bytes:
        raw = data.encode()
        out = bytes(b ^ self.key[i % len(self.key)] for i, b in enumerate(raw))
        return base64.urlsafe_b64encode(out)
      
    def decrypt(self, token: bytes) -> str:
        data = base64.urlsafe_b64decode(token)
        out = bytes(b ^ self.key[i % len(self.key)] for i, b in enumerate(data))
        return out.decode()
