import random
from typing import Any, Callable, List, Optional, Generic, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Strategy(Generic[T_co]):
    """Base class for property-based strategies."""

    def __class_getitem__(cls, item: Any) -> "Strategy":
        """Allow ``Strategy[...]`` for type hinting purposes."""
        return cls

    def example(self) -> Any:
        raise NotImplementedError

    def map(self, func: Callable[[Any], Any]) -> "Strategy":
        parent = self

        class _Mapped(Strategy):
            def example(self) -> Any:
                return func(parent.example())

        return _Mapped()

    def filter(self, pred: Callable[[Any], bool]) -> "Strategy":
        parent = self

        class _Filtered(Strategy):
            def example(self) -> Any:
                val = parent.example()
                while not pred(val):
                    val = parent.example()
                return val

        return _Filtered()

    def flatmap(self, func: Callable[[Any], "Strategy"]) -> "Strategy":
        parent = self

        class _FlatMapped(Strategy):
            def example(self) -> Any:
                return func(parent.example()).example()

        return _FlatMapped()

    def example_many(self, n: int) -> List[Any]:
        return [self.example() for _ in range(n)]


class FloatStrategy(Strategy[float]):
    """Strategy for generating random floats."""

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        allow_nan: bool = True,
        allow_infinity: bool = True,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_infinity = allow_infinity

    def example(self) -> float:
        return random.uniform(self.min_value, self.max_value)


class IntegerStrategy(Strategy[int]):
    """Strategy for generating random integers."""

    def __init__(self, min_value: int = 0, max_value: int = 100) -> None:
        self.min_value = min_value
        self.max_value = max_value

    def example(self) -> int:
        return random.randint(self.min_value, self.max_value)


class ListStrategy(Strategy[List[Any]]):
    """Strategy for generating lists of elements from another strategy."""

    def __init__(self, element: Strategy, min_size: int = 0, max_size: Optional[int] = None) -> None:
        self.element = element
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else min_size

    def example(self) -> list[Any]:
        if self.max_size == self.min_size:
            size = self.min_size
        else:
            size = random.randint(self.min_size, self.max_size)
        return [self.element.example() for _ in range(size)]


def floats(*, min_value: float = 0.0, max_value: float = 1.0, allow_nan: bool = True, allow_infinity: bool = True) -> FloatStrategy:
    """Create a float strategy."""
    return FloatStrategy(min_value=min_value, max_value=max_value, allow_nan=allow_nan, allow_infinity=allow_infinity)


def integers(min_value: int = 0, max_value: int = 100) -> IntegerStrategy:
    """Create an integer strategy."""
    return IntegerStrategy(min_value=min_value, max_value=max_value)


def lists(element: Strategy[T_co], min_size: int = 0, max_size: Optional[int] = None) -> ListStrategy:
    """Create a list strategy."""
    return ListStrategy(element, min_size=min_size, max_size=max_size)


# Alias for compatibility with Hypothesis' public API
SearchStrategy = Strategy

__all__ = ["floats", "integers", "lists", "Strategy", "SearchStrategy"]
