"""Runtime stubs for :mod:`argon2` used in tests."""


class PasswordHasher:
    """Runtime stub for :class:`argon2.PasswordHasher`."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Accept arbitrary arguments."""

    def hash(self, _password: str, *_args: object, **_kwargs: object) -> str:
        """Return a deterministic hash for testing."""
        return "stub-hash"

    def verify(self, _hashed: str, _password: str, *_args: object, **_kwargs: object) -> bool:
        """Trivially verify hashes for testing."""
        return True
