import functools
import inspect
import random

from . import strategies


class _Strategy:
    def example(self):
        raise NotImplementedError

    def map(self, func):
        parent = self
        class _Mapped(_Strategy):
            def example(self):
                return func(parent.example())
        return _Mapped()


def given(**kwargs):
    def decorator(func):
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args, **fkwargs):
            values = {name: strat.example() for name, strat in kwargs.items()}
            fkwargs.update(values)
            result = func(*args, **fkwargs)
            if inspect.iscoroutine(result):
                import asyncio
                return asyncio.get_event_loop().run_until_complete(result)
            return result

        params = [p for p in sig.parameters.values() if p.name not in kwargs]
        wrapper.__signature__ = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)
        return wrapper
    return decorator


def settings(**kwargs):
    def decorator(func):
        return func
    return decorator

__all__ = ["given", "settings", "strategies"]
