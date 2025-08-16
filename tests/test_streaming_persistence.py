import json
import typing
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from memory_system.api.app import stream_memories
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.store import SQLiteMemoryStore
from memory_system.settings import DatabaseConfig, UnifiedSettings


@pytest.mark.asyncio
async def test_saved_batches_survive_restart(tmp_path):
    np = pytest.importorskip("numpy")
    settings = UnifiedSettings.for_testing()
    settings.database = DatabaseConfig(
        db_path=tmp_path / "memory.db",
        vec_path=tmp_path / "memory.vectors",
        cache_path=tmp_path / "memory.cache",
    )

    dim = settings.model.vector_dim
    emb0 = np.random.rand(dim).astype("float32").tolist()
    emb1 = np.random.rand(dim).astype("float32").tolist()

    async def interrupted() -> typing.AsyncIterator[dict[str, typing.Any]]:
        yield {"text": "first", "embedding": emb0}
        yield {"text": "second", "embedding": emb1}
        raise RuntimeError("boom")

    async with EnhancedMemoryStore(settings) as store:
        with pytest.raises(RuntimeError):
            await store.add_memories_streaming(interrupted(), batch_size=2)

    store2 = EnhancedMemoryStore(settings)
    res = await store2.semantic_search(vector=emb0, k=1)
    assert res and res[0].text == "first"
    assert store2.vector_store.stats().total_vectors == 2
    await store2.close()


@pytest.mark.asyncio
async def test_stream_invalid_json():
    store = SQLiteMemoryStore()
    await store.initialise()

    class Req:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines
            self.app = SimpleNamespace(state=SimpleNamespace(memory_store=store))

        async def stream(self) -> typing.AsyncIterator[bytes]:
            for line in self._lines:
                yield line

    req = Req([b'{"text": "ok"}\n', b"invalid json\n"])
    with pytest.raises(HTTPException) as exc:
        await stream_memories(req)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    await store.aclose()


@pytest.mark.asyncio
async def test_stream_invalid_schema():
    store = SQLiteMemoryStore()
    await store.initialise()

    class Req:
        def __init__(self, line: bytes) -> None:
            self._line = line
            self.app = SimpleNamespace(state=SimpleNamespace(memory_store=store))

        async def stream(self) -> typing.AsyncIterator[bytes]:
            yield self._line

    payload = b"{}\n"
    req = Req(payload)
    with pytest.raises(HTTPException) as exc:
        await stream_memories(req)  # type: ignore[arg-type]
    assert exc.value.status_code == 422
    await store.aclose()


@pytest.mark.asyncio
async def test_stream_text_too_long():
    long_text = "x" * 10_001
    store = SQLiteMemoryStore()
    await store.initialise()

    class Req:
        def __init__(self, line: bytes) -> None:
            self._line = line
            self.app = SimpleNamespace(state=SimpleNamespace(memory_store=store))

        async def stream(self) -> typing.AsyncIterator[bytes]:
            yield self._line

    payload = json.dumps({"text": long_text}).encode() + b"\n"
    req = Req(payload)
    with pytest.raises(HTTPException) as exc:
        await stream_memories(req)  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    await store.aclose()


@pytest.mark.asyncio
async def test_stream_text_too_long_enhanced():
    long_text = "x" * 10_001
    settings = UnifiedSettings.for_testing()
    async with EnhancedMemoryStore(settings) as store:

        class Req:
            def __init__(self, line: bytes) -> None:
                self._line = line
                self.app = SimpleNamespace(state=SimpleNamespace(memory_store=store))

            async def stream(self) -> typing.AsyncIterator[bytes]:
                yield self._line

        payload = json.dumps({"text": long_text}).encode() + b"\n"
        req = Req(payload)
        with pytest.raises(HTTPException) as exc:
            await stream_memories(req)  # type: ignore[arg-type]
        assert exc.value.status_code == 400
