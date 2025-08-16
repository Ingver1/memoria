import builtins
import importlib
import sys

import pytest


@pytest.fixture
def afile_module(monkeypatch):
    """Load afile without aiofiles to trigger the fallback implementation."""

    monkeypatch.delitem(sys.modules, "memory_system.utils.afile", raising=False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aiofiles":
            raise ImportError("simulated import error")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module("memory_system.utils.afile")
    monkeypatch.setattr(builtins, "__import__", original_import)
    return module


@pytest.mark.asyncio
async def test_afile_fallback(afile_module, tmp_path):
    """Ensure afile falls back when aiofiles is unavailable."""

    assert hasattr(afile_module, "_AsyncFile")

    sample = tmp_path / "data.txt"
    sample.write_text("line1\nline2\n")
    async with afile_module.open(str(sample)) as f:
        lines = [line async for line in f]
    assert lines == ["line1\n", "line2\n"]


@pytest.mark.asyncio
async def test_afile_read_write(afile_module, tmp_path):
    """Verify read and write operations work via the fallback implementation."""

    # Reading
    src = tmp_path / "src.txt"
    src.write_text("hello world")
    async with afile_module.open(str(src)) as f:
        data = await f.read()
    assert data == "hello world"

    # Writing
    dest = tmp_path / "dest.txt"
    async with afile_module.open(str(dest), "w") as f:
        await f.write("some text")
    assert dest.read_text() == "some text"
