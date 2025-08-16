"""HTTP related utility types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HTTPTimeouts:
    """HTTP client timeout configuration."""

    connect: float = 5.0
    read: float = 10.0
