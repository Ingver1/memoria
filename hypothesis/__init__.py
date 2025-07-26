import functools
import inspect
import random
from typing import Any, Callable, Protocol, TypeVar, cast
from typing_extensions import TypeGuard

from . import strategies

T = TypeVar('T')

class ConfigurableFunction(Protocol):
    _settings: Any


class _Strategy:
    """Minimal strategy base for property-based tests."""
    def example(self) -> object:
        raise NotImplementedError
    def map(self, func: Callable[[Any], Any]) -> '_Strategy':
        parent = self
        class _Mapped(_Strategy):
            def example(self) -> object:
                return func(parent.example())
        return _Mapped()
    def filter(self, pred: Callable[[Any], bool]) -> '_Strategy':
        parent = self
        class _Filtered(_Strategy):
            def example(self) -> object:
                val = parent.example()
                while not pred(val):
                    val = parent.example()
                return val
        return _Filtered()
    def flatmap(self, func: Callable[[Any], '_Strategy']) -> '_Strategy':
        parent = self
        class _FlatMapped(_Strategy):
            def example(self) -> object:
                return func(parent.example()).example()
        return _FlatMapped()

def reify(strat: _Strategy) -> object:
    """Get a concrete example from a strategy."""
    return strat.example()

def given(**kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for property-based tests. Injects strategy examples."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args: Any, **fkwargs: Any) -> T:
                values = {name: strat.example() for name, strat in kwargs.items()}
                fkwargs.update(values)
                return await cast(Any, func)(*args, **fkwargs)

            wrapper_func: Callable[..., T] = async_wrapper
        else:
            def sync_wrapper(*args: Any, **fkwargs: Any) -> T:
                values = {name: strat.example() for name, strat in kwargs.items()}
                fkwargs.update(values)
                return func(*args, **fkwargs)

            wrapper_func = sync_wrapper

        if inspect.isfunction(wrapper_func):
            params = [p for p in sig.parameters.values() if p.name not in kwargs]
            wrapper_func.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        return cast(Callable[..., T], functools.wraps(func)(wrapper_func))

    return decorator

def settings(**kwargs: Any) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Dummy settings decorator for property-based tests."""
    class Config:
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)
            
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        f = cast(ConfigurableFunction, func)
        f._settings = Config(**kwargs)
        return func
        
    return decorator

__all__ = ["given", "settings", "strategies", "reify"]
