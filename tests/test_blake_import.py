import builtins
import importlib
import sys

import pytest


def test_missing_blake3_raises(monkeypatch):
    monkeypatch.delitem(sys.modules, "memory_system.utils.blake", raising=False)
    monkeypatch.delitem(sys.modules, "blake3", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "blake3":
            msg = "No module named 'blake3'"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc:
        importlib.import_module("memory_system.utils.blake")

    assert "blake3" in str(exc.value)
    assert "ai-memory[full]" in str(exc.value)
