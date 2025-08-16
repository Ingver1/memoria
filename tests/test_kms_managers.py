import base64
import sys
import types

import pytest

from memory_system.utils.security import AWSKMSKeyManager, VaultKeyManager


class DummyHVACClient:
    def __init__(self, url, token):
        assert url == "http://vault.local"
        assert token == "token123"

        class KVv2:
            @staticmethod
            def read_secret_version(path, mount_point="secret"):
                assert path == "my/secret"
                assert mount_point == "secret"
                return {"data": {"data": {"key": "vault-fernet-key"}}}

        class KV:
            v2 = KVv2()

        class Secrets:
            kv = KV()

        self.secrets = Secrets()


def test_vault_key_manager_retrieves_key(monkeypatch):
    hvac_module = types.SimpleNamespace(Client=DummyHVACClient)
    monkeypatch.setitem(sys.modules, "hvac", hvac_module)
    manager = VaultKeyManager(url="http://vault.local", token="token123", secret_path="my/secret")
    assert manager.get_key() == "vault-fernet-key"


def test_vault_key_manager_missing_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "hvac", raising=False)
    manager = VaultKeyManager(url="u", token="t", secret_path="p")
    with pytest.raises(RuntimeError):
        manager.get_key()


class DummyBoto3Client:
    def __init__(self, service_name, **kwargs):
        assert service_name == "kms"
        self.kwargs = kwargs

    def generate_data_key(self, KeyId, KeySpec):
        assert KeyId == "key-123"
        assert KeySpec == "AES_256"
        return {"Plaintext": b"0" * 32}


def test_aws_kms_key_manager_retrieves_key(monkeypatch):
    boto3_module = types.SimpleNamespace(
        client=lambda service_name, **kw: DummyBoto3Client(service_name, **kw)
    )
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    manager = AWSKMSKeyManager(key_id="key-123", region_name="us-east-1")
    expected = base64.urlsafe_b64encode(b"0" * 32).decode()
    assert manager.get_key() == expected


def test_aws_kms_key_manager_missing_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "boto3", raising=False)
    manager = AWSKMSKeyManager(key_id="key-123", region_name="us-east-1")
    with pytest.raises(RuntimeError):
        manager.get_key()
