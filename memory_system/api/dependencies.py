"""dependencies.py — FastAPI dependency helper functions for Unified Memory System."""

from __future__ import annotations

import functools
import logging
import typing

from fastapi import HTTPException, status
from memory_system.config.settings import UnifiedSettings
from memory_system.core.store import EnhancedMemoryStore
from memory_system.core.store import get_store as _core_get_store
from memory_system.utils.security import EnhancedPIIFilter

__all__ = [
    "get_settings",
    "get_memory_store",
    "get_store",
    "get_pii_filter",
    "require_api_enabled",
]

log = logging.getLogger("ums.dependencies")


@functools.lru_cache
def get_settings() -> UnifiedSettings:
    """Provide a cached UnifiedSettings instance (singleton)."""
    return UnifiedSettings()


@functools.lru_cache
def get_memory_store() -> EnhancedMemoryStore:
    """Provide a cached EnhancedMemoryStore instance (singleton)."""
    return EnhancedMemoryStore(get_settings())  # Note: runs in sync for simplicity


@functools.lru_cache
def get_pii_filter() -> EnhancedPIIFilter:
    """Provide a cached PII filter instance."""
    return EnhancedPIIFilter()


def get_store() -> typing.Generator[EnhancedMemoryStore, None, None]:
    """Yield the global memory store for dependency injection."""
    store = get_memory_store()
    yield store


def require_api_enabled(settings: UnifiedSettings | None = None) -> None:
    """FastAPI dependency that raises an HTTP 503 if the API is disabled in settings."""
    settings = settings or get_settings()
    if not settings.api.enable_api:
        log.warning("API is disabled by configuration — blocking request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API is currently disabled by configuration.",
        )
