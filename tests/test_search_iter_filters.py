import asyncio
import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore


def test_search_iter_filters(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "db.sqlite")

    async def _run():
        await store.initialise()
        m1 = Memory.new("a", level=0)
        m2 = Memory.new("b", level=1)
        m3 = Memory.new("c", level=2, metadata={"final": True})
        m4 = Memory.new("d", level=2, metadata={"final": False})
        for m in (m1, m2, m3, m4):
            await store.add(m)

        texts = []
        async for chunk in store.search_iter(min_level=1):
            texts.extend(mem.text for mem in chunk)
        assert set(texts) == {"b", "c", "d"}

        texts = []
        async for chunk in store.search_iter(final=True):
            texts.extend(mem.text for mem in chunk)
        assert texts == ["c"]

        texts = []
        async for chunk in store.search_iter(final=False):
            texts.extend(mem.text for mem in chunk)
        assert set(texts) == {"a", "b", "d"}

        await store.aclose()

    try:
        from memory_system.utils.loop import get_loop

        get_loop().run_until_complete(_run())
    except RuntimeError:
        pytest.skip("No usable event loop available in sandbox")
