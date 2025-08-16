"""
Local embedding helper using SentenceTransformer.

Provides ``embed`` function compatible with ``UnifiedSettings`` integration.

Call :func:`close_models` when shutting down your service to release any
loaded models and their resources.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
import contextlib
from typing import TYPE_CHECKING, Any, Protocol, cast

from memory_system.utils.dependencies import require_numpy, require_sentence_transformers

# Optional language detection helper.  ``detect_langs`` defaults
# to ``None`` when the dependency is not installed so callers can fall back to
# simple heuristics without additional checks.
detect_langs: Callable[[str], list[Any]] | None
try:  # optional language detection
    from langdetect import DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        from langdetect import detect_langs as _detect_langs

        detect_langs = _detect_langs
    except Exception:
        detect_langs = None
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    detect_langs = None
    LangDetectException = Exception


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from collections.abc import Callable

    from numpy import ndarray


class EmbeddingModel(Protocol):
    """Protocol describing the minimal embedding interface."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the size of generated embedding vectors."""

    def encode(self, sentences: list[str]) -> ndarray | list[list[float]]:
        """Return embeddings for ``sentences``."""


_MODEL_CACHE: dict[str, EmbeddingModel] = {}
_model_lock = threading.Lock()
ASCII_MAX = 127
LANG_CONFIDENCE_THRESHOLD = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.5"))

# Re-export for tests that patch the class directly
SentenceTransformer: Callable[..., EmbeddingModel] = require_sentence_transformers()


def _detect_language_conf(text: str) -> tuple[str, float]:
    """Return detected language and confidence."""
    if detect_langs is None:
        lang = "non-en" if any(ord(c) > ASCII_MAX for c in text) else "en"
        return lang, 1.0
    try:
        res = detect_langs(text)
        if res:
            top = res[0]
            return top.lang, float(getattr(top, "prob", 0.0))
    except LangDetectException:  # pragma: no cover - detection failure
        pass
    return "en", 0.0


def _detect_language(text: str) -> str:
    lang, conf = _detect_language_conf(text)
    if conf < LANG_CONFIDENCE_THRESHOLD:
        return "en"
    return lang


def _select_model_name(lang: str) -> tuple[str, str]:
    """Return cache key and model name for ``lang``."""
    if lang != "en":
        name = os.getenv("AI_MODEL__MULTILINGUAL_MODEL_NAME")
        if not name:
            name = os.getenv("AI_MODEL__MODEL_NAME", "all-MiniLM-L6-v2")
        return "multi", name
    name = os.getenv("AI_MODEL__MODEL_NAME", "all-MiniLM-L6-v2")
    return "en", name


def _load_model(name: str) -> EmbeddingModel:
    """Load an embedding model by ``name``."""
    log = logging.getLogger(__name__)
    st = require_sentence_transformers()

    class _StubModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 384

        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

    try:
        try:
            model = cast("EmbeddingModel", st(name))
        except TypeError:
            model = cast("EmbeddingModel", st())
        if not hasattr(model, "encode"):
            model = _StubModel()
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        log.exception("Failed to load embedding model '%s'", name)
        raise RuntimeError(f"Failed to load embedding model '{name}': {exc}") from exc
    return model


def _get_model_for_lang(lang: str) -> EmbeddingModel:
    key, name = _select_model_name(lang)
    with _model_lock:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = _load_model(name)
            _MODEL_CACHE[key] = model
        return model


def embed(text: str | list[str]) -> ndarray:
    """
    Return embeddings for ``text`` without external API calls.

    Empty strings produce zero vectors and are allowed within lists.
    """
    np = require_numpy()
    if isinstance(text, str):
        texts = [text]
    elif isinstance(text, list):
        if not text:
            raise ValueError("text cannot be empty")
        texts = []
        for idx, t in enumerate(text):
            if not isinstance(t, str):
                raise TypeError(f"text[{idx}] must be a string, got {type(t).__name__}")
            texts.append(t)
    else:  # pragma: no cover - defensive programming
        raise TypeError("text must be a string or list of strings")

    log = logging.getLogger(__name__)
    log.info("Embedding batch size %d", len(texts))

    if not texts:
        raise ValueError("text cannot be empty")
    model: EmbeddingModel | None = None
    vecs: ndarray | None = None

    if not any(texts):
        model = _get_model_for_lang("en")
        dim = model.get_sentence_embedding_dimension()
        vecs = np.zeros((len(texts), dim), dtype=np.float32)
        return vecs[0] if len(vecs) == 1 else vecs

    lang_map: dict[str, list[int]] = {}
    for idx, t in enumerate(texts):
        if t:
            lang, conf = _detect_language_conf(t)
            if conf < LANG_CONFIDENCE_THRESHOLD:
                lang = "en"
            lang_map.setdefault(lang, []).append(idx)

    if not lang_map:
        model = _get_model_for_lang("en")
        dim = model.get_sentence_embedding_dimension()
        vecs = np.zeros((len(texts), dim), dtype=np.float32)

    for lang, indices in lang_map.items():
        model = _get_model_for_lang(lang)
        group = [texts[i] for i in indices]
        batch_vecs = cast("ndarray", np.asarray(model.encode(group), dtype=np.float32))
        if vecs is None:
            dim = batch_vecs.shape[1]
            vecs = np.zeros((len(texts), dim), dtype=np.float32)
        vecs[indices] = batch_vecs

    if vecs is None:
        if model is None:
            model = _get_model_for_lang("en")
        dim = model.get_sentence_embedding_dimension()
        vecs = np.zeros((len(texts), dim), dtype=np.float32)

    vecs = cast("ndarray", vecs)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    if len(vecs) == 1:
        return vecs[0]
    return vecs


def close_models() -> None:
    """
    Release any loaded embedding models and free library resources.

    This resets cached model instances so that memory is reclaimed. Call this
    when your application is shutting down to avoid holding on to CPU or GPU
    memory. Models will be lazily reloaded on the next :func:`embed` call.
    """
    models = list(_MODEL_CACHE.values())
    _MODEL_CACHE.clear()

    if models:
        import ctypes
        import gc
        import inspect

        _pyframe_locals_to_fast = cast("Any", ctypes.pythonapi.PyFrame_LocalsToFast)
        _pyframe_locals_to_fast.argtypes = [ctypes.py_object, ctypes.c_int]
        _pyframe_locals_to_fast.restype = None

        for m in models:
            for ref in gc.get_referrers(m):
                if inspect.isframe(ref):
                    locs = ref.f_locals
                    for k, v in list(locs.items()):
                        if v is m:
                            locs[k] = None
                    _pyframe_locals_to_fast(ref, 1)
                elif isinstance(ref, dict):
                    for k, v in list(ref.items()):
                        if v is m:
                            ref[k] = None

        try:  # optional torch cleanup
            import torch

            if getattr(torch, "cuda", None) and torch.cuda.is_available():  # pragma: no cover
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch may be unavailable/broken
            # If torch import fails or raises during initialisation, ignore.
            pass

        del models
        gc.collect()
