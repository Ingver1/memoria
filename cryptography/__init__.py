# Minimal cryptography stub providing Fernet used in tests.
# WARNING: This is NOT secure. Only for testing/mocking purposes.
from __future__ import annotations

import base64
import os
from typing import Union

__all__ = ["fernet"]

class InvalidToken(Exception):
    """Raised when decryption fails."""
    pass

class Fernet:
    """Minimal Fernet-like stub for tests. Not secure!"""
    def __init__(self, key: Union[bytes, str]) -> None:
        """Initialize with a base64-encoded 32-byte key."""
        if isinstance(key, str):
            key = key.encode()
        try:
            decoded = base64.urlsafe_b64decode(key)
        except Exception as exc:
            raise ValueError("Invalid encryption key") from exc
        if len(decoded) != 32:
            raise ValueError("Invalid encryption key")
        self.key = decoded

    @staticmethod
    def generate_key() -> bytes:
        """Generate a random base64-encoded 32-byte key."""
        return base64.urlsafe_b64encode(os.urandom(32))

    def encrypt(self, data: bytes) -> bytes:
        """Fake encrypt: XOR with key, then base64 encode."""
        # Not secure! Just for round-trip testing.
        xored = bytes(b ^ self.key[i % 32] for i, b in enumerate(data))
        return base64.urlsafe_b64encode(xored)

    def decrypt(self, token: bytes) -> bytes:
        """Fake decrypt: base64 decode, then XOR with key."""
        try:
            xored = base64.urlsafe_b64decode(token)
            return bytes(b ^ self.key[i % 32] for i, b in enumerate(xored))
        except Exception as exc:
            raise InvalidToken from exc

    def __repr__(self) -> str:
        key_str = base64.urlsafe_b64encode(self.key)[:8].decode('ascii', errors='replace')
        return f"<Fernet key={key_str}...>"

# Expose in submodule style
class fernet:
    """Submodule-style access for Fernet and InvalidToken."""
    Fernet = Fernet
    InvalidToken = InvalidToken
