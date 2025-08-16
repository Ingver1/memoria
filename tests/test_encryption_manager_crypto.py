import pytest

from tests.conftest import _require_crypto

pytestmark = pytest.mark.needs_crypto
_require_crypto()

from cryptography.fernet import InvalidToken

from memory_system.utils.security import EncryptionManager


def test_encryption_manager_roundtrip() -> None:
    mgr = EncryptionManager()
    text = "secret message"
    token = mgr.encrypt(text)
    assert mgr.decrypt(token) == text


def test_encryption_manager_tampered_token() -> None:
    mgr = EncryptionManager()
    token = mgr.encrypt("secret")
    tampered = token[:-1] + (b"0" if token[-1:] != b"0" else b"1")
    with pytest.raises(InvalidToken):
        mgr.decrypt(tampered)
