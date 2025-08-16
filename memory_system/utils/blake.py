"""
Minimal BLAKE3 wrapper.

This module provides a thin wrapper around the :mod:`blake3` package and raises
an informative error if the dependency is missing.
"""

from __future__ import annotations

from typing import cast

try:
    import blake3 as _blake3
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via tests
    msg = (
        "The 'blake3' package is required. Install it directly or use "
        "'pip install ai-memory[full]'."
    )
    raise ModuleNotFoundError(msg) from exc


def blake3_digest(data: bytes, length: int = 16) -> bytes:
    """Return a raw BLAKE3 digest."""
    return cast("bytes", _blake3.blake3(data).digest(length))


def blake3_hex(data: bytes, length: int = 16) -> str:
    """Return a hexadecimal BLAKE3 digest."""
    return blake3_digest(data, length).hex()
