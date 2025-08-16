import time

import pytest

from memory_system.core.store import Memory, SQLiteMemoryStore, _normalize_for_hash
from memory_system.utils.blake import blake3_digest, blake3_hex


@pytest.mark.asyncio
async def test_normalization_and_upsert(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    meta = {"lang": "en", "source": "user"}
    await store.add(Memory.new("Hello   World", metadata=meta))
    await store.add(Memory.new("  hello world  ", metadata=meta))
    async with store.transaction() as conn:
        cur = await conn.execute("SELECT count(*) FROM memories")
        count = (await cur.fetchone())[0]
        cur = await conn.execute("SELECT access_count FROM memories")
        acount = (await cur.fetchone())[0]
    assert count == 1
    assert acount == 1

    n1 = _normalize_for_hash("Hello   World")
    n2 = _normalize_for_hash("  hello world  ")
    h1 = blake3_hex(f"{n1}|en|user".encode())
    h2 = blake3_hex(f"{n2}|en|user".encode())
    assert h1 == h2


@pytest.mark.asyncio
async def test_lang_source_distinct(tmp_path):
    store = SQLiteMemoryStore(str(tmp_path / "db.sqlite"))
    await store.add(Memory.new("same", metadata={"lang": "en", "source": "a"}))
    await store.add(Memory.new("same", metadata={"lang": "fr", "source": "a"}))
    async with store.transaction() as conn:
        cur = await conn.execute("SELECT count(*) FROM memories")
        count = (await cur.fetchone())[0]
    assert count == 1


def test_blake3_digest_speed():
    data = b"x" * 1_000_000
    start = time.perf_counter_ns()
    blake3_digest(data)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    assert elapsed < 0.5
