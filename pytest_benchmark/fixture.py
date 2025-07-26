"""Fixtures and types for pytest-benchmark stub."""

from typing import Any, Callable


class BenchmarkFixture:
    """Minimal stand-in for the real BenchmarkFixture."""

    def __call__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
