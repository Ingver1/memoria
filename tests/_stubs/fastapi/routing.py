"""Routing utilities for FastAPI stub."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Route:
    """Simple representation of a route used by the test stubs."""

    path: str
    endpoint: Callable[..., Any]
    methods: set[str]


class APIRouter:
    """Very small subset of :class:`fastapi.APIRouter` used in tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple container
        self.routes: list[Route] = []

    # Helper to register a route
    def _add_route(
        self, path: str, methods: list[str], func: Callable[..., Any]
    ) -> Callable[..., Any]:
        self.routes.append(Route(path=path, endpoint=func, methods={m.upper() for m in methods}))
        return func

    def get(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda func: self._add_route(path, ["GET"], func)

    def post(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda func: self._add_route(path, ["POST"], func)

    def delete(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda func: self._add_route(path, ["DELETE"], func)

    def patch(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a handler for HTTP PATCH requests."""
        return lambda func: self._add_route(path, ["PATCH"], func)

    def put(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a handler for HTTP PUT requests."""
        return lambda func: self._add_route(path, ["PUT"], func)

    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        methods: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._add_route(path, methods or [], endpoint)
