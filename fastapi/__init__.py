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


class FastAPI:
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


class APIRouter:
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

    def delete(self, path: str, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            self.routes.append(("DELETE", path, func))
            return func

        return decorator


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail


def Query(default: Any = None, **_: Any) -> Any:
    return default


class Depends:
    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency


class _Status:
    HTTP_201_CREATED = 201
    HTTP_422_UNPROCESSABLE_ENTITY = 422
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_503_SERVICE_UNAVAILABLE = 503
    HTTP_204_NO_CONTENT = 204


status = _Status()
