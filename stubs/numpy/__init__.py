"""Runtime stub for :mod:`numpy` used in tests."""

__stub__ = True
float32 = float
float64 = float
int64 = int

from collections.abc import Sequence
from typing import Any

ndarray = Any


def asarray(obj: Any, dtype: Any | None = None) -> Any: ...


def dot(a: Any, b: Any) -> Any: ...


def stack(arrays: Sequence[Any], axis: int = 0) -> Any: ...


def vstack(arrays: Sequence[Any]) -> Any: ...


def mean(a: Any, axis: int | None = None) -> Any: ...


def sqrt(x: Any) -> Any: ...


class linalg:
    @staticmethod
    def norm(a: Any, axis: int | None = None, keepdims: bool = False) -> Any: ...


def _any(obj: object) -> bool:
    """Return ``bool(obj)``."""
    return bool(obj)


def clip(x: float, a: float, b: float) -> float:
    """Clip ``x`` to the inclusive range [``a``, ``b``]."""
    return max(a, min(b, x))


any = _any
