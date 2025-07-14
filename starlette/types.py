from typing import Any, Awaitable, Callable

from .responses import Response

__all__ = ["ASGIApp", "Request", "Response", "RequestResponseEndpoint"]

ASGIApp = Callable[..., Awaitable[Any]]
Request = Any
RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]
