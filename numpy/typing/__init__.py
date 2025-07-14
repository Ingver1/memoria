from __future__ import annotations

from typing import Any, Generic, List, TypeVar

T = TypeVar("T")

class NDArray(List[T], Generic[T]):
    pass
