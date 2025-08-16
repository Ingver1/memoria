"""
Unified Memory System.

A production-grade long-term memory backend for AI agents, assistants, and
generative-retrieval workloads.

This package provides:
- FastAPI-based REST API for memory management
- Vector search capabilities with FAISS
- Semantic embeddings with sentence-transformers
- Security features (PII filtering, encryption)
- Monitoring and health checks
- Scalable SQLite backend with connection pooling
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from httpx import AsyncClient  # noqa: F401

try:  # pragma: no cover - optional dependency during testing
    import httpx

    ASGITransport = getattr(httpx, "ASGITransport", None)

    if ASGITransport and "app" not in httpx.AsyncClient.__init__.__code__.co_varnames:

        class _AsyncClient(httpx.AsyncClient):
            def __init__(self, *args: Any, app: Any | None = None, **kwargs: Any) -> None:
                if app is not None and ASGITransport is not None:
                    kwargs["transport"] = ASGITransport(app=app)
                super().__init__(*args, **kwargs)

        # Reassign with a runtime subclass so tests can pass FastAPI apps
        setattr(httpx, "AsyncClient", cast("Any", _AsyncClient))  # noqa: B010
except ModuleNotFoundError:  # pragma: no cover - httpx may not be installed
    pass

__version__: str = "1.0.0"
# Rebuild the module docstring to embed the current version.
__doc__ = f"Unified Memory System v{__version__}.\n\n" + __doc__.split("\n", 1)[1]
__author__ = "Enhanced Memory Team"
__email__ = "kayel20221967@gmail.com"
__description__ = "Enterprise-grade memory system with vector search, FastAPI and monitoring"

# Configure default logging (no handlers by default for library use)
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "EmbeddingService",
    "EnhancedMemoryStore",
    "PreferenceReranker",
    "UnifiedSettings",
    "WorkingMemory",
    "__version__",
    "create_app",
    "get_settings",
]


# Lazy attribute access to avoid heavy imports on package import.
def __getattr__(name: str) -> Any:
    """Lazily import and return objects from submodules on attribute access."""
    if name == "UnifiedSettings":
        from memory_system.settings import UnifiedSettings

        return UnifiedSettings
    if name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore

        return EnhancedMemoryStore
    if name == "EmbeddingService":
        # NOTE: Corrected the returned class for EmbeddingService
        from memory_system.core.embedding import EnhancedEmbeddingService

        return EnhancedEmbeddingService
    if name == "create_app":
        from memory_system.api.app import create_app

        return create_app

    if name == "WorkingMemory":
        from memory_system.working_memory import WorkingMemory

        return WorkingMemory
    if name == "get_settings":
        from memory_system.settings import get_settings as _get_settings

        # Provide a quick way to retrieve global settings helper
        return _get_settings
    if name == "PreferenceReranker":
        from memory_system.adapter.reranker import PreferenceReranker

        return PreferenceReranker
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_version_info() -> dict[str, Any]:
    """Get detailed version information of the Unified Memory System package."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
    }
