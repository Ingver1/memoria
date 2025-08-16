#!/usr/bin/env python3
"""Re-embed stored vectors using the configured model."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np

from memory_system.core.embedding import EmbeddingService
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.settings import UnifiedSettings
from memory_system.utils.security import safe_decrypt

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from cryptography.fernet import Fernet

try:  # optional dependency
    from cryptography.fernet import Fernet

    fernet_cls: type[Fernet] | None = Fernet
except ImportError:  # pragma: no cover - optional dependency
    fernet_cls: type[Fernet] | None = None


async def reembed() -> None:
    """Recompute embeddings for all stored memories."""
    settings = UnifiedSettings()
    service = EmbeddingService(settings.model.model_name, settings)
    store = EnhancedMemoryStore(settings)
    try:
        fernet_key: bytes | None = None
        fernet_obj: Fernet | None = None
        if settings.security.encrypt_at_rest:
            if fernet_cls is None:
                msg = "cryptography package is required for encrypted storage"
                raise RuntimeError(msg)
            fernet_key = settings.security.encryption_key.get_secret_value().encode()
            fernet_obj = fernet_cls(fernet_key)

        by_mod: dict[str, dict[str, list[str]]] = {}
        async for chunk in store.meta_store.search_iter(chunk_size=1000):
            for mem in chunk:
                text = mem.text
                if fernet_key and fernet_obj:
                    text = safe_decrypt(text, fernet_key, fernet_obj)
                info = by_mod.setdefault(mem.modality, {"ids": [], "texts": []})
                info["ids"].append(mem.id)
                info["texts"].append(text)

        for mod, info in by_mod.items():
            vectors = await service.embed_text(info["texts"])
            store.vector_store.rebuild(mod, np.asarray(vectors, dtype=np.float32), info["ids"])

        await asyncio.to_thread(store.vector_store.save, str(settings.database.vec_path))
    finally:
        await store.close()
        await service.close()


if __name__ == "__main__":  # pragma: no cover - manual execution
    logging.basicConfig(level=logging.INFO)
    asyncio.run(reembed())
