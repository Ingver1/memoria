"""dependencies.py — FastAPI dependency helper functions for Unified Memory System."""

from __future__ import annotations

import logging
import typing

from fastapi import Depends, HTTPException, Request, status

try:  # pragma: no cover - optional fastapi security modules
    from fastapi.security import OAuth2PasswordBearer, SecurityScopes
except Exception:  # pragma: no cover - stub fallback when fastapi not installed

    class SecurityScopes:  # type: ignore[no-redef]
        def __init__(self, scopes: list[str] | None = None) -> None:
            self.scopes = scopes or []

    class OAuth2PasswordBearer:  # type: ignore[no-redef]
        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            self.auto_error = kwargs.get("auto_error", True)

        async def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> str | None:
            return None


from memory_system.core.embedding import EmbeddingService
from memory_system.core.faiss_vector_store import FaissVectorStore
from memory_system.core.store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings
from memory_system.utils.exceptions import SecurityError
from memory_system.utils.security import EnhancedPIIFilter, SecureTokenManager

__all__ = [
    "get_current_user",
    "get_embedding_service",
    "get_max_text_length",
    "get_memory_store",
    "get_pii_filter",
    "get_settings",
    "get_store",
    "get_token_manager",
    "get_vector_store",
    "oauth2_scheme",
    "require_api_enabled",
]

log = logging.getLogger("ums.dependencies")


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "memory:read": "Read access to memories",
        "memory:write": "Write access to memories",
        "admin": "Administrative actions",
    },
)


def get_settings(request: Request) -> UnifiedSettings:
    """Retrieve ``UnifiedSettings`` from ``app.state``."""
    return typing.cast("UnifiedSettings", request.app.state.settings)


def get_token_manager(
    settings: UnifiedSettings = Depends(get_settings),
) -> SecureTokenManager:
    """Return a ``SecureTokenManager`` using configured secrets."""
    key = settings.security.encryption_key.get_secret_value() or settings.security.api_token
    if len(key) < 32:
        key = key.ljust(32, "_")
    return SecureTokenManager(key)


async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
    token_mgr: SecureTokenManager = Depends(get_token_manager),
) -> dict[str, typing.Any]:
    """Validate *token* and ensure required ``security_scopes`` are present."""
    try:
        payload = token_mgr.verify_token(token)
    except SecurityError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from exc
    token_scopes = set(payload.get("scopes", []))
    required = set(security_scopes.scopes)
    if not required.issubset(token_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return payload


def get_pii_filter(request: Request) -> EnhancedPIIFilter:
    """Retrieve the process-wide PII filter."""
    return typing.cast("EnhancedPIIFilter", request.app.state.pii_filter)


def get_embedding_service(request: Request) -> EmbeddingService | None:
    """Fetch the embedding service instance from ``app.state``."""
    return typing.cast("EmbeddingService | None", request.app.state.embedding_service)


def get_vector_store(request: Request) -> FaissVectorStore | None:
    """Fetch the FAISS vector store from ``app.state``."""
    return typing.cast("FaissVectorStore | None", request.app.state.vector_store)


def get_memory_store(request: Request) -> EnhancedMemoryStore:
    """Fetch the EnhancedMemoryStore from ``app.state``."""
    return typing.cast("EnhancedMemoryStore", request.app.state.memory_store)


def get_store(request: Request) -> typing.Generator[EnhancedMemoryStore]:
    """Yield the request-bound memory store for dependency injection."""
    store = get_memory_store(request)
    yield store


def get_max_text_length(store: typing.Any) -> int:
    """
    Return the maximum allowed text length for the given store.

    This helper mirrors the prior inline ``getattr`` chain used throughout the
    API for retrieving ``settings.security.max_text_length`` while providing a
    default of ``10_000`` when the path does not exist.
    """
    return getattr(
        getattr(getattr(store, "settings", None), "security", None),
        "max_text_length",
        10_000,
    )


def require_api_enabled(request: Request, settings: UnifiedSettings | None = None) -> None:
    """FastAPI dependency that raises an HTTP 503 if the API is disabled."""
    settings = settings or get_settings(request)
    if not settings.api.enable_api:
        log.warning("API is disabled by configuration — blocking request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The API is currently disabled by configuration.",
        )
