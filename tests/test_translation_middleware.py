import asyncio

import pytest
from fastapi.testclient import TestClient

from memory_system.api import middleware as mw
from memory_system.api.app import create_app
from memory_system.settings import APIConfig, UnifiedSettings


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_translation_applied(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(
        enable_translation=True,
        translation_confidence_threshold=0.6,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        monkeypatch.setattr(mw, "_detect_language", lambda text: ("fr", 0.4))
        monkeypatch.setattr(mw, "_translate_to_en", lambda text, source_lang=None: "hello")
        resp = client.post("/api/v1/memory/", json={"text": "bonjour"})
        assert resp.status_code == 201
        mem_id = resp.json()["id"]
        store = client.app.state.store
        records = asyncio.get_event_loop().run_until_complete(store.search())
        mem = next(m for m in records if m.id == mem_id)
        assert mem.text == "hello"
        assert mem.metadata["summary"] == "bonjour"
        assert mem.metadata["summary_en"] == "hello"


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_translation_disabled(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(enable_translation=False)
    app = create_app(settings)
    with TestClient(app) as client:
        monkeypatch.setattr(mw, "_detect_language", lambda text: ("fr", 0.4))

        def fail_translate(text, source_lang=None):  # pragma: no cover - should not run
            raise AssertionError("translation should be disabled")

        monkeypatch.setattr(mw, "_translate_to_en", fail_translate)
        resp = client.post("/api/v1/memory/", json={"text": "bonjour"})
        assert resp.status_code == 201
        mem_id = resp.json()["id"]
        store = client.app.state.store
        records = asyncio.get_event_loop().run_until_complete(store.search())
        mem = next(m for m in records if m.id == mem_id)
        assert mem.text == "bonjour"


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_detection_translation_cached(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(
        enable_translation=True,
        translation_confidence_threshold=0.6,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        mw._LANG_CACHE.clear()
        mw._TRANSLATION_CACHE.clear()
        calls = {"detect": 0, "translate": 0}

        def fake_detect(text):
            calls["detect"] += 1
            return "fr", 0.4

        def fake_translate(text, source_lang=None):
            calls["translate"] += 1
            return "hello"

        monkeypatch.setattr(mw, "_detect_language", fake_detect)
        monkeypatch.setattr(mw, "_translate_to_en", fake_translate)
        resp1 = client.post("/api/v1/memory/", json={"text": "bonjour"})
        resp2 = client.post("/api/v1/memory/", json={"text": "bonjour"})
        assert resp1.status_code == resp2.status_code == 201
        assert calls["detect"] == 1
        assert calls["translate"] == 1


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_cache_canonicalizes_text(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(enable_translation=False)
    app = create_app(settings)
    with TestClient(app) as client:
        mw._LANG_CACHE.clear()
        calls = {"detect": 0}

        def fake_detect(text):
            calls["detect"] += 1
            return "fr", 0.4

        monkeypatch.setattr(mw, "_detect_language", fake_detect)
        client.post("/api/v1/memory/", json={"text": "bonjour"})
        client.post("/api/v1/memory/", json={"text": "  BONJOUR   "})
        assert calls["detect"] == 1


def test_cache_key_includes_languages() -> None:
    k1 = mw._cache_key("Hello", "en", "fr")
    k2 = mw._cache_key("hello", "en", "fr")
    k3 = mw._cache_key("hello", "en", "es")
    assert k1 == k2
    assert k2 != k3


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_cache_disable_option(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(enable_translation=False, translation_cache_size=0)
    app = create_app(settings)
    with TestClient(app) as client:
        mw._LANG_CACHE.clear()
        calls = {"detect": 0}

        def fake_detect(text):
            calls["detect"] += 1
            return "fr", 0.4

        monkeypatch.setattr(mw, "_detect_language", fake_detect)
        client.post("/api/v1/memory/", json={"text": "bonjour"})
        client.post("/api/v1/memory/", json={"text": "bonjour"})
        assert calls["detect"] == 2


@pytest.mark.needs_fastapi
@pytest.mark.needs_httpx
def test_mixed_language_sets_lang_und(monkeypatch) -> None:
    settings = UnifiedSettings.for_testing()
    settings.api = APIConfig(
        enable_translation=True,
        translation_confidence_threshold=0.6,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        monkeypatch.setattr(mw, "_detect_language", lambda text: ("ru", 0.4))
        monkeypatch.setattr(mw, "_translate_to_en", lambda text, source_lang=None: "hello")
        resp = client.post("/api/v1/memory/", json={"text": "hello привет"})
        assert resp.status_code == 201
        mem_id = resp.json()["id"]
        store = client.app.state.store
        records = asyncio.get_event_loop().run_until_complete(store.search())
        mem = next(m for m in records if m.id == mem_id)
        assert mem.metadata["canonical_claim"] == "hello привет"
        assert mem.metadata["canonical_claim_en"] == "hello"
        assert mem.metadata["lang"] == "und"
