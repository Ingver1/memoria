from __future__ import annotations

from typing import Any, Callable, Coroutine, TypeVar

__all__ = [
    "FastAPI",
    "APIRouter",
    "Request",
    "HTTPException",
    "Query",
    "Depends",
    "status",
]

F = TypeVar("F", bound=Callable[..., Any])


def _decorator(func: F) -> F:
    return func


class Request:
    app: Any
    headers: dict[str, str]
    client: Any
    url: Any

def __init__(self, scope: dict[str, Any] | None = None) -> None:
        from types import SimpleNamespace

        scope = scope or {}
        self.headers = {
            k.decode(): v.decode()
            for k, v in scope.get("headers", [])
        }
        client = scope.get("client")
        if client is not None:
            host, port = client
        else:
            host, port = None, None
        self.client = SimpleNamespace(host=host, port=port)
        self.url = SimpleNamespace(path=scope.get("path", ""))
        self.app = scope.get("app")


class FastAPI:
    """Minimal FastAPI stub for testing and mocks."""

    def __init__(
        self,
        *args: Any,
        lifespan: Callable[["FastAPI"], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from types import SimpleNamespace

        self.title = kwargs.get("title", "")
        self.version = kwargs.get("version", "")
        self.state = SimpleNamespace()
        self.dependency_overrides: dict[Any, Any] = {}
        self.routes: list[tuple[str, str, Callable[..., Any]]] = []
        self.events: dict[str, list[Callable[..., Any]]] = {"startup": [], "shutdown": []}
        self.lifespan = lifespan

    def add_middleware(self, middleware_class: Any, *args: Any, **kwargs: Any) -> None:
        middleware_class(self, *args, **kwargs)

    def on_event(self, event: str) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.events.setdefault(event, []).append(func)
            return func

        return decorator

    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("GET", path, func))
            return func

        return decorator

    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("POST", path, func))
            return func

        return decorator

    def put(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("PUT", path, func))
            return func

        return decorator

    def patch(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("PATCH", path, func))
            return func

        return decorator

    def options(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("OPTIONS", path, func))
            return func

        return decorator

    def delete(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("DELETE", path, func))
            return func

        return decorator

    def include_router(self, router: Any, *, prefix: str = "") -> None:
        for method, path, func in getattr(router, "routes", []):
            full = prefix + getattr(router, "prefix", "") + path
            self.routes.append((method, full, func))

    def mount(self, path: str, app: Any) -> None:
        self.routes.append(("MOUNT", path, app))

    def __repr__(self) -> str:
        return f"<FastAPI routes={len(self.routes)} events={list(self.events.keys())}>"


class APIRouter:
    """Minimal APIRouter stub for testing and mocks."""

    def __init__(self, *args: Any, prefix: str = "", **kwargs: Any) -> None:
        self.prefix = prefix
        self.routes: list[tuple[str, str, Callable[..., Any]]] = []

    def get(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("GET", path, func))
            return func

        return decorator

    def post(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("POST", path, func))
            return func

        return decorator

    def put(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("PUT", path, func))
            return func

        return decorator

    def patch(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("PATCH", path, func))
            return func

        return decorator

    def options(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("OPTIONS", path, func))
            return func

        return decorator

    def delete(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("DELETE", path, func))
            return func

        return decorator

    def __repr__(self) -> str:
        return f"<APIRouter prefix={self.prefix} routes={len(self.routes)}>"


class HTTPException(Exception):
    """Minimal HTTPException stub."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail

    def raise_for_status(self) -> None:
        raise self

    def __repr__(self) -> str:
        return f"<HTTPException {self.status_code}: {self.detail}>"


def Query(default: Any = None, **_: Any) -> Any:
    """Stub for FastAPI Query."""
    return default


class Depends:
    """Stub for FastAPI Depends."""

    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency

    def __repr__(self) -> str:
        return f"<Depends {self.dependency}>"


class _Status:
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_204_NO_CONTENT = 204
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_503_SERVICE_UNAVAILABLE = 503


status = _Status()
