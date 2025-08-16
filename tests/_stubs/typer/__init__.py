from __future__ import annotations

from typing import Any


class Exit(SystemExit):
    def __init__(self, code: int = 0) -> None:
        super().__init__(code)


class Typer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def command(self, *args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            return func

        return decorator


def Argument(default: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
    return default


def Option(default: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: N802
    return default


class Context:  # pragma: no cover - simple placeholder
    pass


class CallbackParam:  # pragma: no cover
    pass


class BadParameter(Exception):  # noqa: N818 - pragma: no cover
    pass
