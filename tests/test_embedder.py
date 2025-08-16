import logging

import pytest

np = pytest.importorskip("numpy")

from embedder import _detect_language, _get_model_for_lang, embed


def test_embed_logs_batch_size_and_zero_for_empty_string(caplog):
    dim = _get_model_for_lang("en").get_sentence_embedding_dimension()
    with caplog.at_level(logging.INFO):
        vec = embed("")
    assert np.array_equal(vec, np.zeros(dim, dtype=np.float32))
    assert "Embedding batch size 1" in caplog.text


def test_embed_single_empty_string_in_list_returns_zero():
    dim = _get_model_for_lang("en").get_sentence_embedding_dimension()
    vec = embed([""])
    assert np.array_equal(vec, np.zeros(dim, dtype=np.float32))


def test_embed_multiple_empty_strings_in_list_returns_zero():
    dim = _get_model_for_lang("en").get_sentence_embedding_dimension()
    vecs = embed(["", ""])
    assert np.array_equal(vecs, np.zeros((2, dim), dtype=np.float32))


def test_embed_only_empty_strings_returns_zero_vectors() -> None:
    vecs = embed(["", ""])
    dim = _get_model_for_lang("en").get_sentence_embedding_dimension()
    assert np.array_equal(vecs, np.zeros((2, dim), dtype=np.float32))


def test_embed_empty_list_raises():
    with pytest.raises(ValueError):
        embed([])


def test_embed_non_string_element_raises():
    with pytest.raises(TypeError):
        embed(["hello", 123])


def test_embed_invalid_text_type_raises():
    with pytest.raises(TypeError):
        embed(123)


def test_get_model_failure_is_informative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedder as emb

    orig_cache = emb._MODEL_CACHE.copy()
    emb._MODEL_CACHE.clear()

    def fail_loader(name: str) -> None:
        raise OSError("load problem")

    monkeypatch.setattr(emb, "require_sentence_transformers", lambda: fail_loader)
    monkeypatch.setenv("AI_MODEL__MODEL_NAME", "broken-model")

    with pytest.raises(RuntimeError) as exc:
        emb._get_model_for_lang("en")
    assert "broken-model" in str(exc.value)
    assert "load problem" in str(exc.value)

    emb._MODEL_CACHE.clear()
    emb._MODEL_CACHE.update(orig_cache)


def test_no_english_model_loaded_for_non_english() -> None:
    import embedder as emb

    orig_cache = emb._MODEL_CACHE.copy()
    emb._MODEL_CACHE.clear()

    emb.embed("こんにちは")

    assert "en" not in emb._MODEL_CACHE
    assert "multi" in emb._MODEL_CACHE

    emb._MODEL_CACHE.clear()
    emb._MODEL_CACHE.update(orig_cache)


def test_embed_multilingual_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"en": 0, "multi": 0}

    en_model = _get_model_for_lang("en")
    multi_model = _get_model_for_lang("ja")
    orig_en = en_model.encode
    orig_multi = multi_model.encode if multi_model else None

    def wrap_en(texts):
        calls["en"] += 1
        return orig_en(texts)

    def wrap_multi(texts):
        calls["multi"] += 1
        return orig_multi(texts)  # type: ignore[arg-type]

    monkeypatch.setattr(en_model, "encode", wrap_en)
    if multi_model and orig_multi is not None:
        monkeypatch.setattr(multi_model, "encode", wrap_multi)
        embed("こんにちは")
        assert calls["multi"] == 1
    embed("hello")
    assert calls["en"] >= 1


def test_embed_batches_by_language(monkeypatch: pytest.MonkeyPatch) -> None:
    en_model = _get_model_for_lang("en")
    other_lang = _detect_language("こんにちは")
    other_model = _get_model_for_lang(other_lang)

    expected = np.vstack(
        [
            en_model.encode(["hello"])[0],
            other_model.encode(["こんにちは"])[0],
            en_model.encode(["world"])[0],
        ]
    )

    en_calls: list[list[str]] = []
    other_calls: list[list[str]] = []
    orig_en = en_model.encode
    orig_other = other_model.encode if other_model else None

    def wrap_en(texts):
        en_calls.append(list(texts))
        return orig_en(texts)

    def wrap_other(texts):
        other_calls.append(list(texts))
        return orig_other(texts)  # type: ignore[arg-type]

    monkeypatch.setattr(en_model, "encode", wrap_en)
    if other_model and orig_other is not None:
        monkeypatch.setattr(other_model, "encode", wrap_other)

    vecs = embed(["hello", "こんにちは", "world"])

    assert en_calls == [["hello", "world"]]
    if other_model and orig_other is not None:
        assert other_calls == [["こんにちは"]]
    assert np.allclose(vecs, expected)


def test_embed_concurrent_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    import embedder as emb_mod

    orig_cache = emb_mod._MODEL_CACHE.copy()
    emb_mod._MODEL_CACHE.clear()
    counts: dict[str, int] = {}
    orig_init = emb_mod.SentenceTransformer.__init__

    def tracking_init(self, model_name: str) -> None:  # type: ignore[override]
        time.sleep(0.01)
        counts[model_name] = counts.get(model_name, 0) + 1
        orig_init(self, model_name)

    monkeypatch.setattr(emb_mod.SentenceTransformer, "__init__", tracking_init)
    monkeypatch.setenv(
        "AI_MODEL__MULTILINGUAL_MODEL_NAME",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )

    def run() -> None:
        emb_mod.embed("こんにちは")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run) for _ in range(10)]
        for fut in futures:
            fut.result()

    assert "all-MiniLM-L6-v2" not in counts
    assert counts["paraphrase-multilingual-MiniLM-L12-v2"] == 1
    emb_mod._MODEL_CACHE.clear()
    emb_mod._MODEL_CACHE.update(orig_cache)


def test_embed_mixed_language_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    import embedder as emb

    en_model = _get_model_for_lang("en")
    other_model = _get_model_for_lang("ja")
    calls = {"en": 0, "other": 0}
    orig_en = en_model.encode
    orig_other = other_model.encode if other_model else None

    def fake_detect(text: str) -> tuple[str, float]:
        return "ja", 0.4

    monkeypatch.setattr(emb, "_detect_language_conf", fake_detect)

    def wrap_en(texts):
        calls["en"] += 1
        return orig_en(texts)

    monkeypatch.setattr(en_model, "encode", wrap_en)
    if other_model and orig_other is not None:

        def wrap_other(texts):
            calls["other"] += 1
            return orig_other(texts)

        monkeypatch.setattr(other_model, "encode", wrap_other)

    emb.embed("hello こんにちは")

    assert calls["en"] >= 1
    assert calls["other"] == 0
