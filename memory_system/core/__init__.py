# memory_system/core/__init__.py
"""Core module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "EnhancedMemoryStore",
    "EmbeddingService",
    "FaissHNSWIndex",
    "VectorStore",
    "Memory",
    "HealthComponent",
]


def __getattr__(name: str) -> object:
    if name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore

        return EnhancedMemoryStore
    elif name == "EmbeddingService":
        from memory_system.core.embedding import EmbeddingService

        return EmbeddingService
    elif name == "FaissHNSWIndex":
        from memory_system.core.index import FaissHNSWIndex

        return FaissHNSWIndex
    elif name == "VectorStore":
        from memory_system.core.vector_store import VectorStore

        return VectorStore
    elif name in ("Memory", "HealthComponent"):
        from memory_system.core.store import HealthComponent, Memory

        return locals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
