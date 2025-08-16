import builtins
import importlib

import pytest

from memory_system.utils import security


def test_crypto_required(monkeypatch):
    """safe_encrypt/decrypt should error when ``cryptography`` is missing."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sec = importlib.reload(security)

    with pytest.raises(RuntimeError):
        sec.safe_encrypt("secret", b"0" * 32)
    with pytest.raises(RuntimeError):
        sec.safe_decrypt("token", b"0" * 32)

    # Restore original module for subsequent tests
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(sec)
