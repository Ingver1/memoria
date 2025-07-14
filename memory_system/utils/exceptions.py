"""Comprehensive exception hierarchy for Unified Memory System.

Defines structured exception classes with JSON serialization, context info, and cause chaining.
Also provides helper functions for wrapping exceptions and logging.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = [
    "MemorySystemError",
    "ValidationError",
    "ConfigurationError",
    "StorageError",
    "DatabaseError",
    "EmbeddingError",
    "APIError",
    "RateLimitError",
    "TimeoutError",  # Could subclass asyncio.TimeoutError or similar
    "ResourceError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "wrap_exception",
    "create_validation_error",
    "log_exception",
]

log = logging.getLogger(__name__)
_T = TypeVar("_T", bound="MemorySystemError")


class MemorySystemError(RuntimeError):
    """Base exception class for all Unified Memory System errors."""

    default_code: str = "memory_system_error"

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """Initialize a MemorySystemError with an optional context and cause."""
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}
        self.code: str = self.default_code
        self.ts_utc: dt.datetime = dt.datetime.now(dt.timezone.utc)
        if cause is not None:
            self.__cause__ = cause  # built-in exception chaining

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exception to a dictionary for logging or API output."""
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "timestamp": self.ts_utc.isoformat() + "Z",
        }
        if self.context:
            payload["context"] = self.context
        if self.__cause__ is not None:
            payload["cause"] = {
                "type": type(self.__cause__).__name__,
                "message": str(self.__cause__),
            }
        return payload

    def __str__(self) -> str:
        """Return a JSON representation of the exception for readability."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# Specific exception subclasses


class ValidationError(MemorySystemError):
    """Exception for validation failures (e.g., invalid input data)."""

    default_code = "validation_error"


class ConfigurationError(MemorySystemError):
    """Exception for invalid or missing configuration."""

    default_code = "configuration_error"


class StorageError(MemorySystemError):
    """General storage layer failure (e.g., DB or filesystem)."""

    default_code = "storage_error"


class DatabaseError(StorageError):
    """Database-specific errors (e.g., SQL execution, connection issues)."""

    default_code = "database_error"


class EmbeddingError(MemorySystemError):
    """Errors related to embedding generation or model inference."""

    default_code = "embedding_error"


class APIError(MemorySystemError):
    """Generic API error (for API layer issues)."""

    default_code = "api_error"


class RateLimitError(MemorySystemError):
    """Exceeded rate limiting thresholds."""

    default_code = "rate_limit_error"


class TimeoutError(MemorySystemError):
    """Operation timed out."""

    default_code = "timeout_error"


class ResourceError(MemorySystemError):
    """Resource-related failure (e.g., file not found, disk full)."""

    default_code = "resource_error"


class SecurityError(MemorySystemError):
    """Security violation (e.g., encryption or permission issues)."""

    default_code = "security_error"


class AuthenticationError(SecurityError):
    """Authentication failure."""

    default_code = "authentication_error"


class AuthorizationError(SecurityError):
    """Authorization (permission) failure."""

    default_code = "authorization_error"


# Helper functions


def wrap_exception(
    exc_type: type[_T], message: str, **context: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to wrap exceptions in a given exception type with a message and optional context.
    Usage: @wrap_exception(MyError, "Failure occurred", detail=123)
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                raise exc_type(message, context=context, cause=e) from None

        return wrapper

    return decorator


def create_validation_error(message: str, **context: Any) -> ValidationError:
    """Convenience function to create a ValidationError with context."""
    return ValidationError(message, context=context)


def log_exception(
    error: MemorySystemError, logger: logging.Logger | None = None, level: int = logging.ERROR
) -> None:
    """
    Log an exception in structured JSON form.
    If no logger is provided, uses the local module logger.
    """
    if logger is None:
        logger = log
    # Log the JSON representation of the error for structured logging
    logger.log(level, "%s", json.dumps(error.to_dict(), ensure_ascii=False))
