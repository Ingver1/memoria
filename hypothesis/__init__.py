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
        @functools.wraps(func)
        def wrapper(*args: Any, **fkwargs: Any) -> T:
            values = {name: strat.example() for name, strat in kwargs.items()}
            fkwargs.update(values)
            result = func(*args, **fkwargs)
            return result
        if inspect.isfunction(wrapper):
            params = [p for p in sig.parameters.values() if p.name not in kwargs]
            wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        return wrapper
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
