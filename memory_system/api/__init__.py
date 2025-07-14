"""API module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "create_app",
    "HealthResponse",
    "StatsResponse",
    "MemoryCreate",
    "MemoryRead",
    "MemoryUpdate",
    "MemoryQuery",
]


def __getattr__(name: str) -> object:
    """Lazily import objects from submodules on attribute access."""
    if name == "create_app":
        from memory_system.api.app import create_app

        return create_app
    elif name in {"HealthResponse", "StatsResponse", "MemoryCreate", "MemoryRead", "MemoryUpdate", "MemoryQuery"}:
        from memory_system.api.schemas import (
            HealthResponse,
            MemoryCreate,
            MemoryQuery,
            MemoryRead,
            MemoryUpdate,
            StatsResponse,
        )

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
