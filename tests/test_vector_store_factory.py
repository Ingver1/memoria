import sys
import types
from collections.abc import Sequence
from typing import Any

import pytest

from memory_system.core.vector_store import (
    VectorStoreFactory,
    create_vector_store,
    register_vector_store,
)


@register_vector_store("dummy")
class DummyVectorStore:
    def __init__(self, *, dim: int, **_: Any) -> None:  # pragma: no cover - trivial
        self.dim = dim

    async def add(
        self, vectors: Sequence[list[float]], metadata: Sequence[dict[str, Any]]
    ) -> list[str]:  # pragma: no cover - not used
        return []

    async def search(
        self, vector: list[float], k: int = 5
    ) -> list[tuple[str, float]]:  # pragma: no cover - not used
        return []

    async def delete(self, ids: Sequence[str]) -> None:  # pragma: no cover - not used
        return None

    async def flush(self) -> None:  # pragma: no cover - not used
        return None

    async def close(self) -> None:  # pragma: no cover - not used
        return None


@register_vector_store("dummy2")
class OtherDummyVectorStore(DummyVectorStore):
    pass


def test_dynamic_backend_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "memory_system.core.temp_store"
    module = types.ModuleType("temp_store")

    class TempVectorStore(DummyVectorStore):
        pass

    module.TempVectorStore = TempVectorStore
    sys.modules[module_name] = module

    store = create_vector_store(backend=f"{module_name}:TempVectorStore", dim=3)
    assert isinstance(store, TempVectorStore)


@pytest.mark.asyncio
async def test_factory_runtime_swap() -> None:
    factory = VectorStoreFactory(backend="dummy", dim=3)
    store1 = factory.get()
    assert isinstance(store1, DummyVectorStore)

    store2 = await factory.swap("dummy2")
    assert isinstance(store2, OtherDummyVectorStore)
