"""
AWS KMS-backed key management backend.

The backend stores :class:`~memory_system.utils.security.ManagedKey` entries as
encrypted tags on a customer managed key.  Key data is serialised to JSON,
encrypted via :py:meth:`boto3.client("kms").encrypt` and written as a tag value.
When loading, tag values are decrypted back into ``ManagedKey`` instances.

Example:
-------
>>> backend = AWSKMSBackend(key_id="arn:aws:kms:...:key/123", region_name="us-east-1")
>>> backend.save(managed_key)
>>> backend.load_all()
[ManagedKey(...)]

"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from botocore.exceptions import BotoCoreError, ClientError
else:  # pragma: no cover - optional dependency
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:

        class BotoCoreError(Exception): ...

        class ClientError(Exception): ...


from .security import KeyManagementBackend, ManagedKey, SecretStr

log = logging.getLogger(__name__)


class AWSKMSBackend(KeyManagementBackend):
    """Persist keys using AWS Key Management Service."""

    def __init__(self, key_id: str, region_name: str | None = None, **client_kwargs: Any) -> None:
        try:  # pragma: no cover - handled via tests/mocking
            import boto3
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("boto3 is required for AWSKMSBackend") from exc

        self.key_id = key_id
        self._client = boto3.client("kms", region_name=region_name, **client_kwargs)

    def load_all(self) -> list[ManagedKey]:
        tags: list[dict[str, str]] = []
        marker: str | None = None
        while True:
            params: dict[str, Any] = {"KeyId": self.key_id}
            if marker:
                params["Marker"] = marker
            resp = self._client.list_resource_tags(**params)
            tags.extend(resp.get("Tags", []))
            if not resp.get("Truncated"):
                break
            marker = resp.get("NextMarker")

        keys: list[ManagedKey] = []
        for tag in tags:
            value = tag.get("TagValue")
            tag_key = tag.get("TagKey", "<unknown>")
            if not value:
                continue
            try:
                ciphertext = base64.b64decode(value)
                dec = self._client.decrypt(KeyId=self.key_id, CiphertextBlob=ciphertext)
                plaintext: bytes = dec["Plaintext"]
                data = json.loads(plaintext.decode())
                metadata = data["metadata"]
                metadata["created_at"] = datetime.fromisoformat(metadata["created_at"])
                if metadata.get("expires_at") is not None:
                    metadata["expires_at"] = datetime.fromisoformat(metadata["expires_at"])
                data["fernet_key"] = SecretStr(data["fernet_key"])
                keys.append(ManagedKey.model_validate(data))
            except (BotoCoreError, ClientError, ValueError, binascii.Error) as exc:
                log.warning("Skipping malformed tag %s: %s", tag_key, exc)
                continue
        return keys

    def save(self, key: ManagedKey) -> None:
        payload = json.dumps(key.model_dump(mode="json")).encode()
        enc = self._client.encrypt(KeyId=self.key_id, Plaintext=payload)
        ciphertext: bytes = enc["CiphertextBlob"]
        value = base64.b64encode(ciphertext).decode()
        self._client.tag_resource(
            KeyId=self.key_id,
            Tags=[{"TagKey": key.metadata.key_id, "TagValue": value}],
        )

    def delete(self, key_id: str) -> None:
        self._client.untag_resource(KeyId=self.key_id, TagKeys=[key_id])
