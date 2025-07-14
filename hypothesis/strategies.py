import random
from typing import Any


class Strategy:
    def example(self) -> Any:
        raise NotImplementedError

    def map(self, func):
        parent = self
        class _Mapped(Strategy):
            def example(self):
                return func(parent.example())
        return _Mapped()

class FloatStrategy(Strategy):
    def __init__(self, min_value=0.0, max_value=1.0, allow_nan=True, allow_infinity=True):
        self.min_value = min_value
        self.max_value = max_value

    def example(self):
        return random.uniform(self.min_value, self.max_value)

class ListStrategy(Strategy):
    def __init__(self, element: Strategy, min_size: int = 0, max_size: int | None = None):
        self.element = element
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else min_size

    def example(self):
        size = self.min_size if self.max_size == self.min_size else random.randint(self.min_size, self.max_size)
        return [self.element.example() for _ in range(size)]


def floats(**kwargs) -> FloatStrategy:
    return FloatStrategy(**kwargs)


def lists(element: Strategy, min_size: int = 0, max_size: int | None = None) -> ListStrategy:
    return ListStrategy(element, min_size=min_size, max_size=max_size)

__all__ = ["floats", "lists", "Strategy"]
