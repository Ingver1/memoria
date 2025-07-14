# Minimal cryptography stub providing Fernet used in tests.
from __future__ import annotations

import base64
import os

__all__ = ["fernet"]

class InvalidToken(Exception):
    pass

class Fernet:
    def __init__(self, key: bytes | str) -> None:
        if isinstance(key, str):
            key = key.encode()
        try:
            decoded = base64.urlsafe_b64decode(key)
        except Exception as exc:  # pragma: no cover - invalid base64
            raise ValueError("Invalid encryption key") from exc
        if len(decoded) != 32:
            raise ValueError("Invalid encryption key")
        self.key = key

    @staticmethod
    def generate_key() -> bytes:
        return base64.urlsafe_b64encode(os.urandom(32))

    def encrypt(self, data: bytes) -> bytes:
        return base64.urlsafe_b64encode(data)

    def decrypt(self, token: bytes) -> bytes:
        try:
            return base64.urlsafe_b64decode(token)
        except Exception as exc:  # pragma: no cover - simple stub
            raise InvalidToken from exc

# Expose in submodule style
class fernet:
    Fernet = Fernet
    InvalidToken = InvalidToken
