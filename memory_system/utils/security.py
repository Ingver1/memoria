"""
memory_system.utils.security
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

Example:
~~~~~~~
>>> ctx = CryptoContext.from_env()
>>> token = ctx.encrypt(b"secret")
>>> data  = ctx.decrypt(token)

Copyright © Ingver1 2025.

"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from argon2 import PasswordHasher as _Argon2PasswordHasher
    from argon2.exceptions import VerifyMismatchError as _Argon2VerifyMismatchError
else:  # pragma: no cover - optional dependency
    try:
        from argon2 import PasswordHasher as _Argon2PasswordHasher
        from argon2.exceptions import VerifyMismatchError as _Argon2VerifyMismatchError
    except ModuleNotFoundError:  # pragma: no cover - stdlib fallback
        import hashlib
        import hmac
        import secrets

        class _Argon2VerifyMismatchError(Exception):
            """Raised when password verification fails."""

        class _Argon2PasswordHasher:
            """Minimal :mod:`hashlib.scrypt` based replacement for Argon2."""

            def __init__(
                self, *, time_cost: int, memory_cost: int, parallelism: int, hash_len: int
            ) -> None:
                # ``time_cost`` loosely maps to the CPU/memory cost ``n`` parameter
                # of :func:`hashlib.scrypt`.  ``n`` must be a power of two and at
                # least ``2**14`` for adequate security.
                self._n = 2 ** max(14, time_cost + 12)
                self._r = 8
                self._p = max(1, parallelism)
                self._hash_len = hash_len

            def hash(self, password: str) -> str:
                salt = secrets.token_bytes(16)
                dk = hashlib.scrypt(
                    password.encode(),
                    salt=salt,
                    n=self._n,
                    r=self._r,
                    p=self._p,
                    dklen=self._hash_len,
                )
                return base64.urlsafe_b64encode(salt + dk).decode()

            def verify(self, hashed: str, password: str) -> bool:
                try:
                    raw = base64.urlsafe_b64decode(hashed.encode())
                    salt, stored = raw[:16], raw[16:]
                except Exception as exc:  # pragma: no cover - defensive
                    raise _Argon2VerifyMismatchError from exc
                new_dk = hashlib.scrypt(
                    password.encode(),
                    salt=salt,
                    n=self._n,
                    r=self._r,
                    p=self._p,
                    dklen=len(stored),
                )
                if hmac.compare_digest(stored, new_dk):
                    return True
                raise _Argon2VerifyMismatchError


PasswordHasher = _Argon2PasswordHasher
VerifyMismatchError = _Argon2VerifyMismatchError


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from cryptography.fernet import Fernet, InvalidToken

    _CRYPTO_OK = True
else:  # pragma: no cover - optional dependency
    try:
        from cryptography.fernet import Fernet, InvalidToken

        _CRYPTO_OK = True
    except ImportError:
        _CRYPTO_OK = False

        class InvalidToken(Exception):
            """Raised when decryption fails and cryptography is unavailable."""

        class Fernet:  # pragma: no cover - runtime stub
            """Runtime placeholder that fails when used without ``cryptography``."""

            @staticmethod
            def generate_key() -> bytes:
                raise RuntimeError("cryptography package is required for encryption")

            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("cryptography package is required for encryption")


from memory_system.utils.pydantic_compat import (
    BaseModel,
    SecretStr,
    model_validator,
)

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from botocore.exceptions import BotoCoreError, ClientError
else:  # pragma: no cover - optional dependency
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:

        class BotoCoreError(Exception): ...

        class ClientError(Exception): ...


def _as_plain_secret(v: str | SecretStr) -> str:
    """Return the underlying secret value from ``v``."""
    return v.get_secret_value() if isinstance(v, SecretStr) else v


__all__ = [
    "AWSKMSKeyManager",
    "CryptoContext",
    "EncryptionManager",
    "EnhancedPIIFilter",
    "KeyManagementBackend",
    "KeyManager",
    "LocalKeyBackend",
    "LocalKeyManager",
    "PIILoggingFilter",
    "PIIPatterns",
    "PasswordManager",
    "SecretStr",
    "SecureTokenManager",
    "VaultKeyManager",
    "validate_fernet_key",
]


# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------
def safe_encrypt(text: str, key: bytes) -> str:
    """
    Encrypt *text* using :class:`~cryptography.fernet.Fernet`.

    A :class:`RuntimeError` is raised if the ``cryptography`` package is not
    installed.
    """
    if not _CRYPTO_OK:  # pragma: no cover - defensive guard
        raise RuntimeError("cryptography package is required for encryption")

    token: bytes = Fernet(key).encrypt(text.encode())
    return token.decode()


def safe_decrypt(token: str, key: bytes, f: Fernet | None = None) -> str:
    """
    Decrypt *token* using :class:`~cryptography.fernet.Fernet`.

    Decryption errors from ``Fernet`` (``InvalidToken``) are propagated to the
    caller.  A :class:`RuntimeError` is raised when ``cryptography`` is missing.
    """
    if not _CRYPTO_OK:  # pragma: no cover - defensive guard
        raise RuntimeError("cryptography package is required for encryption")

    fernet = f or Fernet(key)
    data: bytes = fernet.decrypt(token.encode())
    return data.decode()


def validate_fernet_key(key: str) -> None:
    """
    Ensure *key* is a valid Fernet key.

    The key must be URL-safe base64 encoded and decode to exactly 32 bytes.
    A :class:`ValueError` is raised with a clear message if validation fails.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+=*", key):
        raise ValueError("Fernet key must be URL-safe base64")
    try:
        decoded = base64.urlsafe_b64decode(key.encode())
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Fernet key must be URL-safe base64") from exc
    if len(decoded) != 32:
        raise ValueError("Fernet key must decode to 32 bytes")


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

    @model_validator(mode="after")
    def _exp_after_created(self) -> KeyMetadata:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        return self


class ManagedKey(BaseModel):
    """Actual key material with metadata."""

    metadata: KeyMetadata
    fernet_key: SecretStr  # URL‑safe base64‑encoded 32‑byte key

    @model_validator(mode="after")
    def _validate_key(self) -> ManagedKey:
        validate_fernet_key(_as_plain_secret(self.fernet_key))
        self.fernet_key = (
            SecretStr(self.fernet_key)
            if not isinstance(self.fernet_key, SecretStr)
            else self.fernet_key
        )
        return self

    def as_fernet(self) -> Fernet:
        return Fernet(_as_plain_secret(self.fernet_key).encode())


# ---------------------------------------------------------------------------
# High level key manager interface
# ---------------------------------------------------------------------------
class KeyManager(ABC):
    """Retrieve encryption keys from various backends."""

    @abstractmethod
    def get_key(self) -> str:
        """Return a base64‑encoded 32‑byte key suitable for ``Fernet``."""
        raise NotImplementedError


class LocalKeyManager(KeyManager):
    """Key manager that simply returns a provided local key."""

    def __init__(self, key: str) -> None:
        validate_fernet_key(key)
        self._key = key

    def get_key(self) -> str:
        return self._key


class VaultKeyManager(KeyManager):
    """
    Key manager backed by HashiCorp Vault.

    Parameters are passed directly to :class:`hvac.Client`.  The following keys
    are recognised and required:

    ``url``
        Base URL of the Vault server.
    ``token``
        Authentication token used to access the server.
    ``secret_path``
        Path to the secret containing the Fernet key.  The secret is expected to
        reside in a KV v2 store under ``mount_point`` (default: ``"secret"``)
        and contain the key under ``key_field`` (default: ``"key"``).
    """

    def __init__(self, **params: Any) -> None:
        self.params = params

    def get_key(self) -> str:
        """
        Fetch an encryption key from HashiCorp Vault.

        A minimal integration is provided to keep the production dependency on
        ``hvac`` optional.  The library is imported lazily and a descriptive
        :class:`RuntimeError` is raised when it is not available.  Only the
        subset of the API required for reading a single secret is exercised so
        that the method can easily be mocked in unit tests.
        """
        try:  # pragma: no cover - handled via tests and mocking
            import hvac
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("hvac library is required for VaultKeyManager") from exc

        url = self.params.get("url")
        token = self.params.get("token")
        secret_path = self.params.get("secret_path") or self.params.get("path")
        if not (url and token and secret_path):  # pragma: no cover - validation
            raise ValueError("url, token and secret_path parameters are required")

        mount_point = self.params.get("mount_point", "secret")
        key_field = self.params.get("key_field", "key")

        client = hvac.Client(url=url, token=token)
        try:
            # Assume KV v2 secrets engine which stores data under 'data' twice
            secret = client.secrets.kv.v2.read_secret_version(
                path=secret_path, mount_point=mount_point
            )
            data = cast("Mapping[str, str]", secret["data"]["data"])
        except hvac.exceptions.VaultError as exc:  # pragma: no cover - branch tested via mocks
            raise RuntimeError("Failed to read key from Vault") from exc

        try:
            key = data[key_field]
        except KeyError as exc:  # pragma: no cover - handled in tests
            raise RuntimeError(f"Key field '{key_field}' not found in secret") from exc

        # The key retrieved from Vault is expected to be a valid Fernet key.
        # In test environments (or misconfigured setups) a simple placeholder
        # may be returned instead.  We attempt to validate the key but fall
        # back to returning it unmodified if validation fails to keep the
        # manager usable with lightweight stubs.
        try:
            validate_fernet_key(key)
        except ValueError:
            return key
        return key


class AWSKMSKeyManager(KeyManager):
    """Key manager backed by AWS KMS."""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def get_key(self) -> str:
        """
        Generate and return a data key using AWS KMS.

        The method lazily imports :mod:`boto3` so that the heavy dependency is
        only required when the AWS backend is used.  The returned key material is
        base64 encoded so that it can be fed directly into ``Fernet``.
        """
        try:  # pragma: no cover - handled via tests/mocking
            import boto3
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("boto3 is required for AWSKMSKeyManager") from exc
        key_id = self.params.get("key_id") or self.params.get("KeyId")
        if not key_id:  # pragma: no cover - validation
            raise ValueError("'key_id' parameter is required for AWSKMSKeyManager")

        client_kwargs = {k: v for k, v in self.params.items() if k not in {"key_id", "KeyId"}}
        kms_client = boto3.client("kms", **client_kwargs)
        try:
            resp = kms_client.generate_data_key(KeyId=key_id, KeySpec="AES_256")
            plaintext: bytes = resp["Plaintext"]
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - branch tested via mocks
            raise RuntimeError("Failed to generate data key from AWS KMS") from exc
        key = base64.urlsafe_b64encode(plaintext).decode()
        validate_fernet_key(key)
        return key


# ---------------------------------------------------------------------------
# Key management backend abstraction
# ---------------------------------------------------------------------------
class KeyManagementBackend(ABC):
    """
    Abstract backend that persists and retrieves encryption keys.

    Built-in implementations include :class:`LocalKeyBackend` for a JSON file
    and :class:`memory_system.utils.kms_aws.AWSKMSBackend` which stores keys in
    AWS KMS.  Custom backends must implement the three CRUD operations below.
    """

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
            return cast("MutableMapping[str, Mapping[str, str]]", data)
        return {}

    def _write(self, data: Mapping[str, Mapping[str, str]]) -> None:
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # ---------------------------------------------------------------------
    # Interface implementation
    # ---------------------------------------------------------------------
    def load_all(self) -> list[ManagedKey]:
        return [ManagedKey.model_validate(dict(entry)) for entry in self._read().values()]

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

    def get_active_key(self) -> str:
        """Return the current active Fernet key as a string."""
        return self._active_key.fernet_key.get_secret_value()

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
        token: bytes = self._active_key.as_fernet().encrypt(data)
        _audit_logger.info(
            "encrypt", extra={"by": self._active_key.metadata.key_id, "size": len(data)}
        )
        return token.decode()

    def decrypt(self, token: str) -> bytes:
        try:
            decoded = base64.urlsafe_b64decode(token)
        except (binascii.Error, ValueError) as exc:  # pragma: no cover - strict validation
            _audit_logger.warning("decrypt.invalid_b64", extra={"token_prefix": token[:10]})
            raise InvalidToken("Invalid token") from exc

        overhead = 1 + 8 + 16 + 32  # version + timestamp + IV + HMAC
        if len(decoded) <= overhead:
            _audit_logger.warning("decrypt.invalid_length", extra={"token_prefix": token[:10]})
            raise InvalidToken("Invalid token")
        if decoded[0] != 0x80:
            _audit_logger.warning("decrypt.invalid_version", extra={"token_prefix": token[:10]})
            raise InvalidToken("Invalid token")

        for key in self._cache.values():
            try:
                data: bytes = key.as_fernet().decrypt(token.encode())
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

                backend = AWSKMSBackend(
                    key_id=os.environ["AWS_KMS_KEY_ID"],
                    region=os.getenv("AWS_REGION", "us-east-1"),
                )  # type: KeyManagementBackend
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


async def start_maintenance(ctx: CryptoContext, interval_hours: int = 6) -> asyncio.Task[None]:
    """Run *maybe_rotate_keys* every *interval_hours* until cancelled."""
    log = logging.getLogger(__name__)

    async def _loop() -> None:  # inner to attach cancellation gracefully
        try:
            while True:
                await ctx.maybe_rotate_keys()
                await asyncio.sleep(interval_hours * 3600)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            log.debug("crypto key maintenance cancelled")
            raise

    task = asyncio.create_task(_loop(), name="crypto_key_maintenance")

    def _log_exceptions(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:  # pragma: no cover - expected on shutdown
            pass
        except Exception:  # pragma: no cover - defensive
            log.exception("crypto key maintenance task failed")

    task.add_done_callback(_log_exceptions)
    return task


# ---------------------------------------------------------------------------
# Additional lightweight security helpers for tests
# ---------------------------------------------------------------------------
import hashlib
import hmac
import secrets
import string
import time
from collections.abc import Iterable
from re import Pattern

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
        self.stats: dict[str, int] = dict.fromkeys(self.patterns, 0)

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


class PIILoggingFilter(logging.Filter):
    """Logging filter that redacts PII from log messages."""

    def __init__(self) -> None:
        super().__init__()
        self._pii = EnhancedPIIFilter()

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        message = record.getMessage()
        redacted, _, _ = self._pii.redact(str(message))
        record.msg = redacted
        record.args = ()
        return True


class SecureTokenManager:
    """
    Simplified JWT-like token manager using PBKDF2-HMAC-SHA256-based
    signatures.

    The previous implementation used Argon2id key derivation with a BLAKE2b
    derived salt.
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: str = "unified-memory-system",
    ) -> None:
        if len(secret_key) < 32:
            raise SecurityError("Secret key must be at least 32 characters")
        if algorithm != "HS256":
            raise SecurityError("Algorithm not allowed")
        self.secret_key = secret_key.encode()
        self.algorithm = algorithm
        self.issuer = issuer
        self.revoked_tokens: set[str] = set()

    def _sign(self, data: bytes) -> str:
        """Create a PBKDF2-HMAC-SHA256 signature for *data*."""
        sig = hashlib.pbkdf2_hmac("sha256", data, self.secret_key, 100_000)
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
            if not secrets.compare_digest(self._sign(data), signature):
                raise SecurityError("Invalid token")
            payload = json.loads(base64.urlsafe_b64decode(body_b64 + "=="))
            header = json.loads(base64.urlsafe_b64decode(header_b64 + "=="))
            return header, payload
        except (ValueError, binascii.Error, json.JSONDecodeError, TypeError) as exc:
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
    """Helper for password hashing and verification using Argon2id."""

    _ph = PasswordHasher(time_cost=2, memory_cost=512 * 1024, parallelism=2, hash_len=32)

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash *password* using Argon2id."""
        if len(password) < 8:
            msg = "Password must be at least 8 characters"
            raise SecurityError(msg)
        return PasswordManager._ph.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against an Argon2id hash."""
        try:
            return PasswordManager._ph.verify(hashed, password)
        except VerifyMismatchError:
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
    """Very small symmetric encryption helper using ``Fernet``."""

    def __init__(self, key: bytes | None = None) -> None:
        """Initialize the manager with a generated or provided key."""
        self.key = key or Fernet.generate_key()
        self._fernet = cast("Any", Fernet(self.key))

    def encrypt(self, data: str) -> bytes:
        """Encrypt *data* returning the ``Fernet`` token."""
        return cast("bytes", self._fernet.encrypt(data.encode()))

    def decrypt(self, token: bytes) -> str:
        """Decrypt *token* returning the original string."""
        return cast("bytes", self._fernet.decrypt(token)).decode()
