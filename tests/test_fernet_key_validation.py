import base64
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_system.settings import SecurityConfig
from memory_system.utils.security import (
    KeyMetadata,
    LocalKeyManager,
    ManagedKey,
    SecretStr,
    validate_fernet_key,
)

VALID_KEY = base64.urlsafe_b64encode(b"0" * 32).decode()
INVALID_KEY = base64.urlsafe_b64encode(b"1" * 16).decode()
WRONG_ALPHABET_KEY = "+" + VALID_KEY[1:]


def test_validate_fernet_key_accepts_valid_key() -> None:
    validate_fernet_key(VALID_KEY)


def test_validate_fernet_key_rejects_malformed_base64() -> None:
    with pytest.raises(ValueError):
        validate_fernet_key("not-base64")


def test_validate_fernet_key_rejects_wrong_alphabet() -> None:
    with pytest.raises(ValueError):
        validate_fernet_key(WRONG_ALPHABET_KEY)


def test_validate_fernet_key_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        validate_fernet_key(INVALID_KEY)


def test_security_config_encryption_key_validation() -> None:
    with pytest.raises(ValidationError, match="URL-safe base64"):
        SecurityConfig(encryption_key=WRONG_ALPHABET_KEY)
    with pytest.raises(ValidationError, match="32-byte"):
        SecurityConfig(encryption_key=INVALID_KEY)


def test_managed_key_and_local_manager_validation() -> None:
    meta = KeyMetadata(key_id="id", created_at=datetime.now(UTC))
    with pytest.raises(ValidationError):
        ManagedKey(metadata=meta, fernet_key=SecretStr(INVALID_KEY))
    with pytest.raises(ValueError):
        LocalKeyManager("bad-key")
    # Should not raise for valid key
    LocalKeyManager(VALID_KEY)
