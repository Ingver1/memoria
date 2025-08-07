"""
Ensures `security.encrypt_at_rest=True` really hides plaintext
inside the SQLite backing file (uses SQLCipher driver).
"""

from pathlib import Path

import pytest

from memory_system.config.settings import DatabaseConfig, UnifiedSettings
from memory_system.core.enhanced_store import EnhancedMemoryStore


@pytest.mark.asyncio
async def test_sqlcipher_encryption(tmp_path: Path) -> None:
    """Test that data is properly encrypted at rest when using SQLCipher."""
    db_file = tmp_path / "cipher.db"
    cfg = UnifiedSettings.for_testing()
    # Configure database with SQLCipher and encryption
    cfg.database = DatabaseConfig(db_path=db_file, connection_string=f"sqlite+sqlcipher:///{db_file}")
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})

    store = EnhancedMemoryStore(cfg)
    await store.add_memory(text="secret-string", embedding=[0.0] * cfg.model.vector_dim)
    await store.close()

    # Read raw bytes—plaintext must NOT appear in the file.
    assert b"secret-string" not in db_file.read_bytes()
