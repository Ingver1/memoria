"""Request-scoped context variables."""

from __future__ import annotations

from contextvars import ContextVar

# Context variable storing the current request identifier.
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)

__all__ = ["REQUEST_ID"]
