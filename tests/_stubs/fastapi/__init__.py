"""Minimal FastAPI stub for environments without the real dependency."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from . import routing
from .routing import APIRouter


class HTTPException(Exception):
    def __init__(self, status_code: int = 500, detail: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FastAPI:
    """Minimal subset of :class:`fastapi.FastAPI` used in tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple container
        self.state = SimpleNamespace()
        self.routes: list[routing.Route] = []
        # FastAPI exposes ``dependency_overrides`` for overriding dependencies
        # during tests.  ``create_app`` accesses this attribute during the
        # application start-up event, so the stub provides it as a simple dict.
        self.dependency_overrides: dict[Any, Any] = {}

    # --- Router registration helpers -------------------------------------
    # The real FastAPI application exposes methods like ``get``/``post`` that
    # return decorators used to register path handlers.  The tests only need
    # the decorator behaviour (no actual routing), so each method returns a
    # no-op wrapper that leaves the function unchanged.

    def add_middleware(self, *args: Any, **kwargs: Any) -> None:
        return None

    def middleware(
        self, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def _add_route(
        self, path: str, methods: list[str], func: Callable[..., Any]
    ) -> Callable[..., Any]:
        self.routes.append(
            routing.Route(path=path, endpoint=func, methods={m.upper() for m in methods})
        )
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
        return lambda func: self._add_route(path, ["PATCH"], func)

    def put(
        self, path: str, *args: Any, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda func: self._add_route(path, ["PUT"], func)

    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        methods: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._add_route(path, methods or [], endpoint)

    # Starlette's FastAPI subclass exposes ``mount`` to attach sub-applications
    # (e.g. the metrics endpoint).  The tests only check that the application
    # records the mounted path, so the stub simply stores it as a route with no
    # HTTP methods.
    def mount(self, path: str, app: Any, *args: Any, **kwargs: Any) -> None:
        self.routes.append(routing.Route(path=path, endpoint=app, methods=set()))

    def include_router(self, router: Any, prefix: str = "", *args: Any, **kwargs: Any) -> None:
        for r in getattr(router, "routes", []):
            self.routes.append(
                routing.Route(path=prefix + r.path, endpoint=r.endpoint, methods=set(r.methods))
            )

    def on_event(self, _event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


class Request:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.scope = {}

    async def stream(self) -> Any:  # pragma: no cover - simple async iterator
        if False:
            yield b""  # type: ignore[misc]


class Response:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.status_code = kwargs.get("status_code", 200)


class _Status:
    def __getattr__(self, name: str) -> int:  # pragma: no cover - simple attribute default
        return 0


status = _Status()


def Body(
    default: Any = None,
    *,
    example: Any | None = None,
    examples: Sequence[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return default


def Query(
    default: Any = None,
    *,
    example: Any | None = None,
    examples: Sequence[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return default


def Depends(dep: Any | None = None) -> Any:
    return dep


# Re-export stub submodules
from . import middleware, openapi, testclient  # noqa: E402

__all__ = [
    "APIRouter",
    "Body",
    "Depends",
    "FastAPI",
    "HTTPException",
    "Query",
    "Request",
    "Response",
    "middleware",
    "openapi",
    "routing",
    "status",
    "testclient",
]
