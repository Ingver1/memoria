from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

from memory_system.unified_memory import Memory, _get_cache

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "to",
    "of",
    "in",
    "on",
    "for",
    "by",
    "with",
    "as",
    "at",
    "is",
    "it",
}


def _token_set(s: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(s.lower()) if t not in STOP_WORDS}


def _token_list(s: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(s.lower()) if t not in STOP_WORDS]


ScoreFunc = Callable[[Sequence[Memory], str], Sequence[float]]


def score_token_overlap(memories: Sequence[Memory], query: str) -> list[float]:
    """Count overlapping non-stopword tokens between *query* and each memory."""
    if not memories:
        return []
    cache = _get_cache()
    q = _token_set(query)
    mem_tokens: list[set[str]] = []
    for mem in memories:
        tokens = mem._tokens
        if tokens is None:
            tokens = _token_set(mem.text)
            mem._tokens = tokens
        mem_tokens.append(tokens)
    raw: list[float] = []
    for mem, tokens in zip(memories, mem_tokens, strict=True):
        key = f"attn_tok:{query}|{mem.memory_id}"
        cached = cache.get(key)
        if cached is not None:
            raw.append(float(cached))
            continue
        score = float(len(q & tokens))
        cache.put(key, score)
        raw.append(score)
    return raw


def _hash_embedding(text: str, dim: int = 32) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vals = [b / 255.0 for b in h]
    if len(vals) < dim:
        reps = (dim + len(vals) - 1) // len(vals)
        vals = (vals * reps)[:dim]
    else:
        vals = vals[:dim]
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def score_embeddings(memories: Sequence[Memory], query: str) -> list[float]:
    """Cosine similarity between hashed embeddings of *query* and *memories*."""
    if not memories:
        return []
    cache = _get_cache()
    q_key = f"attn_emb_q:{query}"
    q_vec = cache.get(q_key)
    if q_vec is None:
        q_vec = _hash_embedding(query)
        cache.put(q_key, q_vec)
    mem_vecs: list[list[float]] = []
    for mem in memories:
        key = f"attn_emb:{mem.memory_id}"
        vec = cache.get(key)
        if vec is None:
            vec = _hash_embedding(mem.text)
            cache.put(key, vec)
        mem_vecs.append(vec)
    return [sum(q * v for q, v in zip(q_vec, vec, strict=False)) for vec in mem_vecs]


def score_tfidf(memories: Sequence[Memory], query: str) -> list[float]:
    """Simple TF-IDF scoring between *query* and *memories*."""
    if not memories:
        return []
    docs = [_token_list(mem.text) for mem in memories]
    df: dict[str, int] = {}
    for tokens in docs:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    n = len(memories)
    idf = {t: math.log(n / (1 + df_t)) for t, df_t in df.items()}
    q_tokens = _token_list(query)
    scores: list[float] = []
    for tokens in docs:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_tokens:
            s += tf.get(t, 0) * idf.get(t, 0.0)
        scores.append(s)
    return scores


def attention_weights(
    memories: Sequence[Memory],
    query: str,
    *,
    scoring: ScoreFunc | None = None,
    temperature: float = 1.0,
    weight_key: str | None = None,
) -> list[float]:
    """
    Return attention weights for *memories* given a *query*.

    ``weight_key`` may reference a field inside ``Memory.metadata`` whose value
    influences ranking. The scoring callable returns raw similarity scores which
    are combined with any metadata weight and memory ``importance`` attributes
    before applying the softmax. If *scoring* is not provided, token overlap
    scoring is used.
    """
    if not memories:
        return []

    if scoring is None:
        scoring = score_token_overlap

    raw = list(scoring(memories, query))

    meta_weights = [
        float((mem.metadata or {}).get(weight_key, 0.0)) if weight_key else 0.0 for mem in memories
    ]
    priors = [getattr(mem, "importance", 0.0) or 0.0 for mem in memories]
    scores = [r + mw + p for r, mw, p in zip(raw, meta_weights, priors, strict=True)]
    max_score = max(scores) if scores else 0.0
    exps = [math.exp((s - max_score) / max(1e-6, temperature)) for s in scores]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def order_by_attention(
    memories: Sequence[Memory],
    query: str,
    *,
    scoring: ScoreFunc | None = None,
    **kw: Any,
) -> list[Memory]:
    w = attention_weights(memories, query, scoring=scoring, **kw)
    return [m for m, _ in sorted(zip(memories, w, strict=True), key=lambda x: x[1], reverse=True)]


__all__ = [
    "attention_weights",
    "order_by_attention",
    "score_embeddings",
    "score_tfidf",
    "score_token_overlap",
]
