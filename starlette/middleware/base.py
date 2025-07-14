from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..types import ASGIApp, Request, RequestResponseEndpoint, Response

__all__ = ["BaseHTTPMiddleware", "RequestResponseEndpoint"]


class BaseHTTPMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        return await call_next(request)
