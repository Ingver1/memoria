import base64
import logging
import sys
import types
from datetime import UTC, datetime

from memory_system.utils.kms_aws import AWSKMSBackend
from memory_system.utils.security import KeyMetadata, ManagedKey, SecretStr


class DummyKMSClient:
    def __init__(self, region_name=None, **kwargs):
        self.kwargs = {"region_name": region_name, **kwargs}
        self.tags: dict[str, str] = {}

    def encrypt(self, KeyId, Plaintext):
        return {"CiphertextBlob": base64.b64encode(Plaintext)}

    def decrypt(self, CiphertextBlob, KeyId):
        return {"Plaintext": base64.b64decode(CiphertextBlob)}

    def tag_resource(self, KeyId, Tags):
        for tag in Tags:
            self.tags[tag["TagKey"]] = tag["TagValue"]

    def list_resource_tags(self, KeyId, Marker=None):
        tags = [{"TagKey": k, "TagValue": v} for k, v in self.tags.items()]
        return {"Tags": tags, "Truncated": False}

    def untag_resource(self, KeyId, TagKeys):
        for key in TagKeys:
            self.tags.pop(key, None)


def test_aws_kms_backend_crud(monkeypatch):
    captured: dict[str, DummyKMSClient] = {}

    def client_factory(service_name, region_name=None, **kwargs):
        assert service_name == "kms"
        captured["client"] = DummyKMSClient(region_name=region_name, **kwargs)
        return captured["client"]

    boto3_module = types.SimpleNamespace(client=client_factory)
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)

    backend = AWSKMSBackend(
        key_id="key-123",
        region_name="us-east-1",
        aws_access_key_id="id",
        aws_secret_access_key="secret",
    )

    client = captured["client"]
    assert client.kwargs["region_name"] == "us-east-1"
    assert client.kwargs["aws_access_key_id"] == "id"
    assert client.kwargs["aws_secret_access_key"] == "secret"

    key = ManagedKey(
        metadata=KeyMetadata(key_id="abc", created_at=datetime.now(UTC)),
        fernet_key=SecretStr(base64.urlsafe_b64encode(b"0" * 32).decode()),
    )

    backend.save(key)
    loaded = backend.load_all()
    assert [k.model_dump() for k in loaded] == [key.model_dump()]
    backend.delete("abc")
    assert backend.load_all() == []


def test_aws_kms_backend_skips_malformed_tag(monkeypatch, caplog):
    captured: dict[str, DummyKMSClient] = {}

    def client_factory(service_name, region_name=None, **kwargs):
        assert service_name == "kms"
        captured["client"] = DummyKMSClient(region_name=region_name, **kwargs)
        return captured["client"]

    boto3_module = types.SimpleNamespace(client=client_factory)
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)

    backend = AWSKMSBackend(key_id="key-123", region_name="us-east-1")

    client = captured["client"]
    valid_key = ManagedKey(
        metadata=KeyMetadata(key_id="valid", created_at=datetime.now(UTC)),
        fernet_key=SecretStr(base64.urlsafe_b64encode(b"0" * 32).decode()),
    )
    backend.save(valid_key)

    client.tags["bad"] = base64.b64encode(b"not-json").decode()

    with caplog.at_level(logging.WARNING):
        loaded = backend.load_all()

    assert [k.metadata.key_id for k in loaded] == ["valid"]
    assert any("bad" in record.getMessage() for record in caplog.records)
