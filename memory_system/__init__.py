"""Unified Memory System.

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
from typing import Any

try:  # pragma: no cover - optional dependency during testing
    import httpx
    from httpx import ASGITransport

    if "app" not in httpx.AsyncClient.__init__.__code__.co_varnames:

        class _AsyncClient(httpx.AsyncClient):
            def __init__(self, *args: Any, app: Any | None = None, **kwargs: Any) -> None:
                if app is not None:
                    kwargs["transport"] = ASGITransport(app=app)
                super().__init__(*args, **kwargs)

        # Reassign with a runtime subclass so tests can pass FastAPI apps
        httpx.AsyncClient = _AsyncClient  # type: ignore[misc]
except Exception:  # pragma: no cover - httpx may not be installed
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
    "__version__",
    "UnifiedSettings",
    "EnhancedMemoryStore",
    "EmbeddingService",
    "create_app",
    "get_settings",
]


# Lazy attribute access to avoid heavy imports on package import.
def __getattr__(name: str) -> Any:
    """Lazily import and return objects from submodules on attribute access."""
    if name == "UnifiedSettings":
        from memory_system.config.settings import UnifiedSettings

        return UnifiedSettings
    elif name == "EnhancedMemoryStore":
        from memory_system.core.store import EnhancedMemoryStore

        return EnhancedMemoryStore
    elif name == "EmbeddingService":
        # NOTE: Corrected the returned class for EmbeddingService
        from memory_system.core.embedding import EnhancedEmbeddingService

        return EnhancedEmbeddingService
    elif name == "create_app":
        from memory_system.api.app import create_app

        return create_app
    elif name == "get_settings":
        from memory_system.config.settings import UnifiedSettings

        # Provide a quick way to retrieve global settings
        return UnifiedSettings  # Could also return a get_settings function if defined
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_version_info() -> dict[str, Any]:
    """Get detailed version information of the Unified Memory System package."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": __description__,
    }
