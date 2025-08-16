from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder

    from memory_system.core.store import Memory

logger = logging.getLogger(__name__)

_MODEL: CrossEncoder | None = None


def _get_provider() -> str:
    """
    Return the configured cross‑encoder provider.

    Settings take precedence over the ``CROSS_ENCODER_PROVIDER`` environment
    variable.  Defaults to ``"sbert"`` which uses the local
    ``sentence-transformers`` model.
    """
    try:  # pragma: no cover - optional dependency
        from memory_system.settings import get_settings

        provider = getattr(get_settings().ranking, "cross_encoder_provider", None)
        if provider:
            return str(provider).lower()
    except Exception:  # pragma: no cover - settings not available
        pass
    return os.getenv("CROSS_ENCODER_PROVIDER", "sbert").lower()


def _load_model() -> CrossEncoder | None:
    """
    Load the MiniLM cross-encoder lazily.

    The cross encoder is an optional and fairly heavy dependency.  For most
    deployments we don't want to attempt loading it unless explicitly
    requested.  Users may opt-in by setting the ``ENABLE_CROSS_ENCODER``
    environment variable to a truthy value.  When the variable is absent the
    function simply returns ``None`` which causes callers to skip reranking and
    keep the original ordering.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    flag = os.getenv("ENABLE_CROSS_ENCODER", "")
    if flag.lower() not in {"1", "true", "yes"}:
        logger.warning("cross-encoder disabled; set ENABLE_CROSS_ENCODER=1 to enable")
        _MODEL = None
        return None

    try:
        from sentence_transformers import CrossEncoder
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
        logger.warning("sentence-transformers not available; cross-encoder disabled")
        _MODEL = None
        return None
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("unexpected error importing sentence-transformers: %s", exc)
        raise
    device = "cpu"
    try:  # pragma: no cover - best effort GPU detection
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - torch missing
        pass
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("unexpected error checking torch availability: %s", exc)
        raise
    try:
        _MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    except (OSError, RuntimeError, ImportError) as exc:  # pragma: no cover - model load failure
        logger.warning("failed to load cross-encoder: %s", exc)
        _MODEL = None
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("unexpected error loading cross-encoder: %s", exc)
        raise
    return _MODEL


def _cohere_rerank(memories: Sequence[Memory], query: str) -> list[Memory] | None:
    """
    Rerank using Cohere's hosted cross‑encoder.

    Returns ``None`` when the API or package is unavailable.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY not set; Cohere reranker disabled")
        return None
    try:  # pragma: no cover - optional dependency
        import cohere
    except (ModuleNotFoundError, ImportError):
        logger.warning("cohere package not available; Cohere reranker disabled")
        return None
    model = os.getenv("COHERE_RERANK_MODEL", "rerank-multilingual-v3.5")
    docs = [m.text for m in memories]
    try:
        client = cohere.Client(api_key)
        resp = client.rerank(model=model, query=query, documents=docs)
        scores = [r.relevance_score for r in resp.results]
    except Exception as exc:  # pragma: no cover - API failure
        logger.warning("Cohere rerank failed: %s", exc)
        return None
    ranked = sorted(zip(memories, scores, strict=False), key=lambda x: x[1], reverse=True)
    result: list[Memory] = []
    for mem, score in ranked:
        meta = dict(mem.metadata or {})
        meta["cross_score"] = float(score)
        try:
            mem = replace(mem, metadata=meta)
        except Exception:  # pragma: no cover - replacement failed
            pass
        result.append(mem)
    return result


def order_by_cross_encoder(memories: Sequence[Memory], query: str, **_: object) -> list[Memory]:
    """
    Rerank *memories* by relevance to *query* using a cross-encoder.

    When the model or its dependencies are unavailable the input is returned
    unchanged so callers can depend on this function without heavy installs.
    """
    provider = _get_provider()
    if provider == "cohere":
        res = _cohere_rerank(memories, query)
        if res is not None:
            return res
        return list(memories)

    model = _load_model()
    if model is None:
        return list(memories)
    pairs = [(query, m.text) for m in memories]
    try:
        scores = model.predict(pairs)
    except RuntimeError as exc:  # pragma: no cover - inference failure
        logger.warning("cross-encoder inference failed: %s", exc)
        return list(memories)
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("unexpected error during cross-encoder inference: %s", exc)
        raise
    ranked = sorted(zip(memories, scores, strict=False), key=lambda x: x[1], reverse=True)
    result: list[Memory] = []
    for mem, score in ranked:
        meta = dict(mem.metadata or {})
        meta["cross_score"] = float(score)
        try:
            mem = replace(mem, metadata=meta)
        except Exception:  # pragma: no cover - replacement failed
            pass
        result.append(mem)
    return result


__all__ = ["order_by_cross_encoder"]
