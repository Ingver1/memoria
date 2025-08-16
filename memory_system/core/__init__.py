# memory_system/core/__init__.py
"""Core module for Unified Memory System."""

from __future__ import annotations

__all__ = [
    "EmbeddingService",
    "EnhancedMemoryStore",
    "PostgresMemoryStore",
    "FaissHNSWIndex",
    "HealthComponent",
    "Memory",
    "VectorStore",
]


def __getattr__(name: str) -> object:
    if name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore

        return EnhancedMemoryStore
    if name == "EmbeddingService":
        from memory_system.core.embedding import EmbeddingService

        return EmbeddingService
    if name == "FaissHNSWIndex":
        from memory_system.core.index import FaissHNSWIndex

        return FaissHNSWIndex
    if name == "VectorStore":
        from memory_system.core.vector_store import VectorStore

        return VectorStore
    if name == "PostgresMemoryStore":
        from memory_system.core.postgres_store import PostgresMemoryStore

        return PostgresMemoryStore
    if name in ("Memory", "HealthComponent"):
        from memory_system.core.store import HealthComponent, Memory

        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
