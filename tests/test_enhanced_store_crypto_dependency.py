import builtins
import importlib
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_store_operates_without_crypto(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import memory_system.core.enhanced_store as es

    es = importlib.reload(es)

    assert not es.FERNET_AVAILABLE

    class DummyVectorStore:
        def __init__(self, settings) -> None:
            self._stats = SimpleNamespace(total_vectors=0)

        def stats(self):
            return self._stats

        def save(self, path) -> None:
            return None

        def rebuild(self, modality, vecs, ids):
            return None

    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(
        es,
        "require_numpy",
        lambda: SimpleNamespace(asarray=lambda x, dtype=None: x, float32=float),
    )

    from memory_system.settings import UnifiedSettings

    cfg = UnifiedSettings.for_testing()
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": False})

    store = es.EnhancedMemoryStore(cfg)
    await store.start()
    await store.close()


@pytest.mark.asyncio
async def test_store_errors_when_encryption_enabled(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import memory_system.core.enhanced_store as es

    es = importlib.reload(es)

    class DummyVectorStore:
        def __init__(self, settings) -> None:
            self._stats = SimpleNamespace(total_vectors=0)

        def stats(self):
            return self._stats

        def save(self, path) -> None:
            return None

        def rebuild(self, modality, vecs, ids):
            return None

    monkeypatch.setattr(es, "FaissVectorStore", DummyVectorStore)
    monkeypatch.setattr(
        es,
        "require_numpy",
        lambda: SimpleNamespace(asarray=lambda x, dtype=None: x, float32=float),
    )

    from memory_system.settings import UnifiedSettings

    cfg = UnifiedSettings.for_testing()
    cfg.security = cfg.security.model_copy(update={"encrypt_at_rest": True})

    store = es.EnhancedMemoryStore(cfg)
    with pytest.raises(RuntimeError):
        await store.start()
    await store.close()
