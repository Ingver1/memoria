"""
Ensures `security.encrypt_at_rest=True` really hides plaintext
inside the SQLite backing file (uses SQLCipher driver).
"""

import sqlite3
from pathlib import Path

import pytest

from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import DatabaseConfig, UnifiedSettings
from tests.conftest import _require_crypto

pytestmark = pytest.mark.needs_crypto
_require_crypto()


def _read_without_key(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("SELECT count(*) FROM memories")


def _open_with_key(path: Path, key: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA key = ?", (key,))
        conn.execute("SELECT count(*) FROM memories")


@pytest.mark.asyncio
async def test_sqlcipher_encryption(tmp_path: Path) -> None:
    """Test that data is properly encrypted at rest when using SQLCipher."""
    db_file = tmp_path / "cipher.db"
    cfg = UnifiedSettings.for_testing()
    # Configure database with SQLCipher and encryption
    cfg.database = DatabaseConfig(db_path=db_file, url=f"sqlite+sqlcipher:///{db_file}")
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})

    async with EnhancedMemoryStore(cfg) as store:
        await store.add_memory(text="secret-string", embedding=[0.0] * cfg.model.vector_dim)

    # Read raw bytes—plaintext must NOT appear in the file.
    assert b"secret-string" not in db_file.read_bytes()


@pytest.mark.asyncio
async def test_rebuild_index_uses_single_fernet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ensure only one Fernet instance is created during rebuild_index."""
    db_file = tmp_path / "cipher.db"
    vec_file = tmp_path / "vec"  # vector store path
    cache_file = tmp_path / "cache"
    cfg = UnifiedSettings.for_testing()
    cfg.database = DatabaseConfig(db_path=db_file, vec_path=vec_file, cache_path=cache_file)
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})

    async with EnhancedMemoryStore(cfg) as store:
        await store.add_memory(text="secret", embedding=[0.0] * cfg.model.vector_dim)

        from memory_system.core import enhanced_store as es

        calls: list[bytes] = []

        class TrackingFernet(es.Fernet):  # type: ignore[override]
            def __init__(self, key: bytes) -> None:
                calls.append(key)
                super().__init__(key)

        monkeypatch.setattr(es, "fernet_cls", TrackingFernet)
        await store.rebuild_index()

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cannot_read_without_key(tmp_path: Path) -> None:
    """Database should be unreadable without the SQLCipher key."""
    db_file = tmp_path / "cipher.db"
    cfg = UnifiedSettings.for_testing()
    cfg.database = DatabaseConfig(db_path=db_file, url=f"sqlite+sqlcipher:///{db_file}")
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})

    async with EnhancedMemoryStore(cfg) as store:
        await store.add_memory(text="secret", embedding=[0.0] * cfg.model.vector_dim)

    with pytest.raises(sqlite3.DatabaseError):
        _read_without_key(db_file)


@pytest.mark.asyncio
async def test_key_rotation(tmp_path: Path) -> None:
    """Rotating the encryption key should invalidate the old key."""
    db_file = tmp_path / "cipher.db"
    cfg = UnifiedSettings.for_testing()
    cfg.database = DatabaseConfig(db_path=db_file, url=f"sqlite+sqlcipher:///{db_file}")
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})
    old_key = cfg.security.encryption_key.get_secret_value()

    async with EnhancedMemoryStore(cfg) as store:
        await store.add_memory(text="secret", embedding=[0.0] * cfg.model.vector_dim)
        await store.meta_store.rotate_key("new-secret")

    # Opening with the old key should fail
    with pytest.raises(sqlite3.DatabaseError):
        _open_with_key(db_file, old_key)

    # Opening with the new key should succeed
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA key = 'new-secret'")
        count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        assert count == 1
    finally:
        conn.close()
