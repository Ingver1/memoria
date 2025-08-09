"""Minimal stub implementation of cryptography.fernet for tests."""


class InvalidToken(Exception):
    """Stub invalid token exception."""


class Fernet:
    def __init__(self, key: bytes) -> None:  # pragma: no cover - simple stub
        self.key = key

    @staticmethod
    def generate_key() -> bytes:  # pragma: no cover - simple stub
        return b"0" * 32

    def encrypt(self, data: bytes) -> bytes:  # pragma: no cover - simple stub
        return data

    def decrypt(self, token: bytes) -> bytes:  # pragma: no cover - simple stub
        return token
