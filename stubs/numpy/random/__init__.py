"""Minimal runtime stub for ``numpy.random`` used by mypy.

This module only exists to satisfy type checkers that attempt to resolve
``numpy.random`` and nested modules such as ``numpy.random.mtrand``. It is not a
functional implementation – tests use a lightweight stub under ``tests/_stubs``
when NumPy is not installed.
"""

from __future__ import annotations

from typing import Any


def seed(value: int | None = None) -> None:  # pragma: no cover - stub
    """No-op seed function for compatibility."""
    return None


class Generator:  # pragma: no cover - stub
    pass
