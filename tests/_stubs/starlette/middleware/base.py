"""Middleware base stubs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

RequestResponseEndpoint = Callable[..., Awaitable[Any]]


class BaseHTTPMiddleware:
    def __init__(self, app: Any, *args: Any, **kwargs: Any) -> None:
        self.app = app

    async def dispatch(
        self, request: Any, call_next: RequestResponseEndpoint
    ) -> Any:  # pragma: no cover
        return await call_next(request)

    async def __call__(self, request: Any) -> Any:  # pragma: no cover - simple passthrough
        return await self.dispatch(request, self.app)
