"""AWS KMS-backed key management backend stub.

This module intentionally provides only a placeholder implementation.  The
``AWSKMSBackend`` class exposes the same interface as the local backend but has
no real integration with AWS KMS.  Every method simply raises
``NotImplementedError`` so that accidental use of this class is immediately
obvious at runtime.
"""

from .security import KeyManagementBackend, ManagedKey


class AWSKMSBackend(KeyManagementBackend):
    """Stub implementation for future AWS KMS support."""

    def __init__(self, key_id: str, region: str) -> None:
        """Create a stub backend configured for a specific KMS key.

        Parameters
        ----------
        key_id:
            The identifier or ARN of the KMS key that would be used.
        region:
            AWS region where the key resides.
        """
        self.key_id = key_id
        self.region = region

    def load_all(self) -> list[ManagedKey]:
        """Retrieve all managed keys from KMS (not implemented).

        A real implementation would list and decrypt the stored keys
        associated with ``self.key_id`` via AWS KMS APIs.
        """
        raise NotImplementedError("AWS KMS backend is not implemented")

    def save(self, key: ManagedKey) -> None:
        """Persist a new key to KMS (not implemented).

        The placeholder does not actually upload anything but merely raises
        ``NotImplementedError``.
        """
        raise NotImplementedError("AWS KMS backend is not implemented")

    def delete(self, key_id: str) -> None:
        """Remove a key from KMS (not implemented).

        A production version would schedule the key for deletion using AWS
        KMS.  Here we simply raise ``NotImplementedError``.
        """
        raise NotImplementedError("AWS KMS backend is not implemented")
