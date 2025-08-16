import numpy as np
import pytest
from cryptography.fernet import Fernet

from memory_system.core.maintenance import forget_old_memories
from memory_system.core.store import ENVELOPE_MANAGER, Memory, SQLiteMemoryStore


@pytest.mark.asyncio
async def test_purge_disables_decryption(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()

    class DummyIndex:
        def add_vectors(self, ids, vecs):
            return None

        def remove_ids(self, ids):
            return None

    index = DummyIndex()
    mem = Memory.new("secret")
    await store.add(mem)
    past = __import__("datetime").datetime.now(__import__("datetime").UTC) - __import__(
        "datetime"
    ).timedelta(days=2)
    await store.update_memory(mem.id, metadata={"last_accessed": past.isoformat()})

    async with store.transaction() as conn:
        cur = await conn.execute(
            "SELECT ciphertext, cek_wrapped FROM memories WHERE id=?", (mem.id,)
        )
        row = await cur.fetchone()
    cipher, wrapped = row
    wrapped_bytes = wrapped if isinstance(wrapped, bytes) else wrapped.encode()
    cek = ENVELOPE_MANAGER.decrypt(wrapped_bytes)
    assert Fernet(cek.encode()).decrypt(cipher).decode() == "secret"

    await forget_old_memories(store, index, min_total=0, retain_fraction=1.0, ttl=60.0)

    async with store.transaction() as conn:
        cur = await conn.execute(
            "SELECT ciphertext, cek_wrapped FROM memories WHERE id=?", (mem.id,)
        )
        row = await cur.fetchone()
    _, wrapped = row
    assert wrapped is None
    with pytest.raises(Exception):
        ENVELOPE_MANAGER.decrypt(wrapped)  # type: ignore[arg-type]

    await store.aclose()


@pytest.mark.asyncio
async def test_rebuild_index_skips_purged(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "mem.db")
    await store.initialise()

    class DummyIndex:
        def __init__(self):
            self.ids: set[str] = set()

        def add_vectors(self, ids, vecs):
            self.ids.update(ids)

        def remove_ids(self, ids):
            for i in ids:
                self.ids.discard(i)

    index = DummyIndex()
    mem = Memory.new("hello")
    await store.add(mem)
    index.add_vectors([mem.id], np.zeros((1, 3), dtype=np.float32))
    past = __import__("datetime").datetime.now(__import__("datetime").UTC) - __import__(
        "datetime"
    ).timedelta(days=2)
    await store.update_memory(mem.id, metadata={"last_accessed": past.isoformat()})
    await forget_old_memories(store, index, min_total=0, retain_fraction=1.0, ttl=60.0)

    rebuilt = DummyIndex()
    async for chunk in store.search_iter():
        ids = [m.id for m in chunk]
        rebuilt.add_vectors(ids, np.zeros((len(ids), 3), dtype=np.float32))

    assert mem.id not in rebuilt.ids
    await store.aclose()
