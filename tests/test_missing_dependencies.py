import builtins
import importlib

import pytest

from memory_system.utils import security
from tests import conftest as cfg


@pytest.mark.smoke
def test_fake_embed_skips_without_numpy(monkeypatch):
    """fake_embed fixture should skip when numpy is absent."""
    monkeypatch.setattr(cfg, "np", None, raising=False)
    with pytest.raises(pytest.skip.Exception):
        cfg.fake_embed.__wrapped__(monkeypatch)


@pytest.mark.smoke
def test_require_crypto_skips_without_cryptography(monkeypatch):
    """_require_crypto should skip when cryptography is absent."""
    monkeypatch.setattr(cfg, "_has", lambda m: False, raising=False)
    with pytest.raises(pytest.skip.Exception):
        cfg._require_crypto()


@pytest.mark.smoke
def test_safe_encrypt_errors_without_cryptography(monkeypatch):
    """safe_encrypt should error when cryptography is missing."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(security)
    with pytest.raises(RuntimeError):
        security.safe_encrypt("secret", b"0" * 32)
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(security)
