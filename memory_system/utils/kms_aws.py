"""Placeholder for an AWS KMS-backed key management backend.

This module is intentionally left unimplemented. The :class:`AWSKMSBackend`
class exposes the same interface as the local backend but does not provide a
real integration with AWS KMS. All methods will raise
``NotImplementedError`` when called. This makes the absence of an actual AWS
implementation explicit and avoids silent failures if the class is
accidentally used.
"""

from .security import KeyManagementBackend, ManagedKey


class AWSKMSBackend(KeyManagementBackend):
    """Stub implementation for future AWS KMS support."""

    def __init__(self, key_id: str, region: str) -> None:
        """Create a stub backend configured for a specific KMS key."""
        self.key_id = key_id
        self.region = region

    def load_all(self) -> list[ManagedKey]:
        """Retrieve all managed keys from KMS (not implemented)."""
        raise NotImplementedError("AWS KMS backend is not implemented")

    def save(self, key: ManagedKey) -> None:
        """Persist a new key to KMS (not implemented)."""
        raise NotImplementedError("AWS KMS backend is not implemented")

    def delete(self, key_id: str) -> None:
        """Remove a key from KMS (not implemented)."""
        raise NotImplementedError("AWS KMS backend is not implemented")
