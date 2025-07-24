import functools
import inspect
import random
from typing import Callable, Any

from . import strategies


class _Strategy:
    def example(self) -> object:
        raise NotImplementedError

    def map(self, func: Callable[[Any], Any]) -> '_Strategy':
        parent = self
        class _Mapped(_Strategy):
            def example(self) -> object:
                return func(parent.example())
        return _Mapped()


def given(**kwargs: Any) -> Callable[..., object]:
    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **fkwargs: Any) -> object:
            values = {name: strat.example() for name, strat in kwargs.items()}
            fkwargs.update(values)
            result = func(*args, **fkwargs)
            if inspect.iscoroutine(result):
                from memory_system.utils.loop import get_or_create_loop
                return get_or_create_loop().run_until_complete(result)
            return result

        # Only assign __signature__ if wrapper is a function (not a class)
        if inspect.isfunction(wrapper):
            params = [p for p in sig.parameters.values() if p.name not in kwargs]
            wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        return wrapper
    return decorator


def settings(**kwargs: Any) -> Callable[..., object]:
    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        return func
    return decorator

__all__ = ["given", "settings", "strategies"]
