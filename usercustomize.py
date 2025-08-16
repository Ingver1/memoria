"""Project-level runtime customizations for tests and compatibility.

This module is automatically imported by Python at startup (if present on
``sys.path``), after the standard ``sitecustomize``. We keep it minimal and
defensive to avoid side effects outside of tests.
"""

from __future__ import annotations

import os as _os

# Ensure pytest doesn't auto-load external plugins that can affect tests.
_os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def _patch_schemathesis() -> None:
    """Provide a stable alias for Schemathesis data generation.

    Older and newer Schemathesis versions expose different attribute names.
    Tests expect ``DataGenerationMethod.fuzzed``; map it to ``positive`` when
    available, otherwise to ``negative``.
    """

    try:
        pass  # type: ignore
    except Exception:
        return

    try:
        from schemathesis import DataGenerationMethod as _DGM  # type: ignore
    except Exception:
        try:
            from schemathesis.specs.openapi import DataGenerationMethod as _DGM  # type: ignore
        except Exception:
            return

    if hasattr(_DGM, "fuzzed"):
        return

    alias = getattr(_DGM, "positive", None) or getattr(_DGM, "negative", None)
    if alias is not None:
        try:
            _DGM.fuzzed = alias
        except Exception:
            # Best-effort patch; ignore if it fails in this environment.
            pass


_patch_schemathesis()
