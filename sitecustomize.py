"""Customise Python runtime for tests and compatibility."""

import asyncio
import builtins as _builtins
import contextlib
import os
import sys as _sys
from pathlib import Path as _Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

# Ensure a default event loop is available for tests that call
# ``asyncio.get_event_loop()`` from synchronous contexts.
try:  # pragma: no cover - defensive initialisation
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Provide a soft fallback for libraries that still use ``get_running_loop`` from
# synchronous code by delegating to ``get_event_loop`` when no loop is running.
_original_grl = asyncio.get_running_loop


def _grl_or_loop() -> asyncio.AbstractEventLoop:  # pragma: no cover - trivial wrapper
    try:
        return _original_grl()
    except RuntimeError:
        return asyncio.get_event_loop()


def _patch_asyncio_get_running_loop() -> None:
    asyncio.get_running_loop = cast("Callable[[], asyncio.AbstractEventLoop]", _grl_or_loop)  # type: ignore[misc, assignment]


_patch_asyncio_get_running_loop()

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


# Python 3.12's ``Path.readlink`` returns a :class:`Path` object, but some tests
# expect the older string return type.  Normalising here keeps compatibility.
def _readlink(self: _Path) -> str:
    return os.readlink(self)


def _patch_path_readlink() -> None:
    _Path.readlink = cast("Callable[[_Path], str]", _readlink)  # type: ignore[misc, assignment]


_patch_path_readlink()

# Ensure numpy stub exposes float64 for tests that reference it


def _patch_numpy() -> None:  # pragma: no cover - simple helper
    np = _sys.modules.get("numpy")
    if np is not None and getattr(np, "__stub__", False):
        np.float64 = cast("type[float]", getattr(np, "float32", float))  # type: ignore[misc, assignment]


def _patch_schemathesis() -> None:  # pragma: no cover - simple helper
    """Provide compatibility shims for different Schemathesis versions.

    - Add ``DataGenerationMethod.fuzzed`` alias when missing.
    """
    sm = _sys.modules.get("schemathesis")
    if sm is None:
        return
    # Try both old and new import paths
    try:
        from schemathesis import DataGenerationMethod  # type: ignore
    except Exception:  # pragma: no cover - fallback for new layout
        try:
            from schemathesis.specs.openapi import DataGenerationMethod  # type: ignore
        except Exception:
            return
    if not hasattr(DataGenerationMethod, "fuzzed"):
        # Prefer positive strategy when available
        alias = getattr(DataGenerationMethod, "positive", None) or getattr(
            DataGenerationMethod, "negative", None
        )
        if alias is not None:
            with contextlib.suppress(Exception):
                DataGenerationMethod.fuzzed = alias


_orig_import = _builtins.__import__


def _import_hook(
    name: str,
    globals_: dict[str, object] | None = None,
    locals_: dict[str, object] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> "ModuleType":  # pragma: no cover
    module = _orig_import(name, globals_, locals_, fromlist, level)
    if name == "numpy" or name.startswith("numpy."):
        _patch_numpy()
    if name == "schemathesis" or name.startswith("schemathesis."):
        _patch_schemathesis()
    return module


def _patch_import_hook() -> None:
    _builtins.__import__ = cast("Callable[..., ModuleType]", _import_hook)  # type: ignore[misc, assignment]


_patch_import_hook()
