"""Minimal stub implementation of cryptography.fernet for tests."""


class InvalidToken(Exception):
    """Stub invalid token exception."""


class Fernet:
    def __init__(self, key: bytes) -> None:  # pragma: no cover - simple stub
        self.key = key

    @staticmethod
    def generate_key() -> bytes:  # pragma: no cover - simple stub
        # The real ``cryptography`` package returns a URL-safe base64 encoded
        # value that decodes to 32 bytes.  The original stub returned raw zero
        # bytes which caused validation errors in tests that expect a valid
        # Fernet key.  Encoding the zeros ensures ``SecurityConfig``'s
        # ``_validate_key`` check accepts the generated key while keeping the
        # implementation lightweight.
        import base64

        return base64.urlsafe_b64encode(b"0" * 32)

    def encrypt(self, data: bytes) -> bytes:  # pragma: no cover - simple stub
        return data

    def decrypt(self, token: bytes) -> bytes:  # pragma: no cover - simple stub
        return token
