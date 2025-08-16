import builtins
import importlib
import logging
import threading
from typing import Any

import pytest


def test_rprint_warning_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name.startswith("rich"):
            raise ModuleNotFoundError
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import memory_system.cli as cli_module

    cli = importlib.reload(cli_module)

    caplog.set_level(logging.WARNING, logger="memoria.cli")

    def call_rprint() -> None:
        cli.rprint("x")

    threads = [threading.Thread(target=call_rprint) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    warnings = [r for r in caplog.records if "Install extras for colours" in r.message]
    assert len(warnings) == 1

    importlib.reload(cli_module)
