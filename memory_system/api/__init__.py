"""API module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "HealthResponse",
    "MemoryCreate",
    "MemoryQuery",
    "MemoryRead",
    "MemoryUpdate",
    "StatsResponse",
    "create_app",
]


def __getattr__(name: str) -> object:
    """Lazily import objects from submodules on attribute access."""
    if name == "create_app":
        from memory_system.api.app import create_app

        return create_app
    if name in {
        "HealthResponse",
        "StatsResponse",
        "MemoryCreate",
        "MemoryRead",
        "MemoryUpdate",
        "MemoryQuery",
    }:
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
