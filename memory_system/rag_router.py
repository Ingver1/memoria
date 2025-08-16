# ruff: noqa: RUF002
"""
RAG routing and retrieval pipeline for Memoria.

This module provides a minimal, production-ready retrieval layer that can
select between multiple memory channels and rank documents using a composite
score. Heavy ML dependencies are loaded lazily so the pipeline still works in
lightweight environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from embedder import LANG_CONFIDENCE_THRESHOLD, _detect_language_conf
from memory_system.core.bandit import rerank_with_bandit
from memory_system.core.mmr import mmr
from memory_system.graph_rag import extract_triples, graph_store
from memory_system.router import Channel
from memory_system.settings import get_settings
from memory_system.utils.dependencies import (
    require_faiss,
    require_httpx,
    require_numpy,
    require_sentence_transformers,
)
from memory_system.utils.http import HTTPTimeouts

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx
    from numpy import ndarray
else:
    httpx = require_httpx()
np = require_numpy()
try:  # optional heavy deps
    ST: Any | None = require_sentence_transformers()
    from sentence_transformers import CrossEncoder as _CrossEncoder
except Exception:  # pragma: no cover
    ST = None
    _CrossEncoder = None
CrossEncoder = _CrossEncoder

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient as _QdrantClient, models as _qdrant_models
except ImportError:  # pragma: no cover - dependency missing
    _QdrantClient = None
    _qdrant_models = None
QdrantClient = _QdrantClient
models = _qdrant_models

logger = logging.getLogger("memoria.rag_router")

ASCII_THRESHOLD = 127
MIN_QUERY_LEN = 6
EPSILON = 1e-9


def _lang_alphabet_mismatch(text: str) -> bool:
    """Return True when detected language and script disagree."""
    lang, conf = _detect_language_conf(text)
    if conf < LANG_CONFIDENCE_THRESHOLD:
        return False
    has_non_ascii = any(ord(c) > ASCII_THRESHOLD for c in text)
    lang_is_en = lang.startswith("en")
    return (has_non_ascii and lang_is_en) or (not has_non_ascii and not lang_is_en)


def calc_entropy(probs: Sequence[float]) -> float:
    """
    Return Shannon entropy for ``probs``.

    Args:
        probs: Sequence of probabilities summing approximately to one.

    Returns:
        Entropy value in nats.

    """

    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * math.log(p)
    return ent


@dataclass
class MemoryDoc:
    """Document returned from the memory service."""

    id: str
    text: str
    channel: Channel
    score_sim: float = 0.0
    score_rerank: float = 0.0
    score_composite: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_ts: float | None = None
    last_access_ts: float | None = None
    access_count: int | None = None
    personalness: float = 0.0
    globalness: float = 0.0


class LocalPersonalStore:
    """In-memory FAISS store for personal memories with optional Qdrant offload."""

    def __init__(
        self,
        dim: int,
        *,
        threshold: int = 10000,
        qdrant_url: str | None = None,
        collection: str = "personal",
        api_key: str | None = None,
    ) -> None:
        faiss = require_faiss()
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim
        self.docs: list[MemoryDoc] = []
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.doc_map: dict[str, MemoryDoc] = {}
        self.vector_map: dict[str, ndarray] = {}
        self._threshold = threshold
        self._use_qdrant = False
        self._qdrant_client = None
        self._qdrant_collection = collection
        self._qdrant_access_counts: dict[
            str, int
        ] = {}  # Improvement: track usage counts for Qdrant-offloaded docs
        if qdrant_url and QdrantClient is not None and models is not None:
            try:
                self._qdrant_client = QdrantClient(url=qdrant_url, api_key=api_key)
                self._qdrant_models = models
            except Exception:  # pragma: no cover - best effort
                logger.debug("Qdrant client not available", exc_info=True)
                self._qdrant_client = None
                self._qdrant_models = None
        else:
            self._qdrant_models = None

    def _init_docs(self, docs: list[MemoryDoc], ts: float) -> None:
        for d in docs:
            d.created_at_ts = d.created_at_ts or ts
            d.last_access_ts = d.last_access_ts or ts
            d.access_count = d.access_count or 0
            d.personalness = d.personalness or 1.0
            d.globalness = d.globalness or 0.0

    def _handle_vectors(self, docs: list[MemoryDoc], vectors: ndarray) -> ndarray:
        vectors = vectors.astype(np.float32, copy=False)
        self.index.add(vectors)
        self.docs.extend(docs)
        self.vectors = np.vstack([self.vectors, vectors])
        for d, v in zip(docs, vectors, strict=False):
            self.doc_map[d.id] = d
            self.vector_map[d.id] = v
            try:
                graph_store.add_card(d.id, d.text)
            except Exception:  # pragma: no cover - indexing should not break storage
                logger.debug("failed to index doc in graph", exc_info=True)
        return vectors

    def _offload_to_qdrant(self, docs: list[MemoryDoc], vectors: ndarray) -> None:
        if (
            self._qdrant_client is None
            or self._qdrant_models is None
            or (len(self.docs) < self._threshold and not self._use_qdrant)
        ):
            return
        models = self._qdrant_models
        qclient = self._qdrant_client
        assert models is not None and qclient is not None
        docs_src = self.docs if not self._use_qdrant else docs
        vecs_src = [self.vector_map[d.id] for d in docs_src] if not self._use_qdrant else vectors
        if not self._use_qdrant and not qclient.collection_exists(
            collection_name=self._qdrant_collection
        ):
            qclient.create_collection(
                collection_name=self._qdrant_collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )
        points = [
            models.PointStruct(
                id=d.id,
                vector=vec.tolist(),
                payload={
                    "text": d.text,
                    "channel": d.channel.value,
                    "metadata": d.metadata,
                    "created_at_ts": d.created_at_ts,
                    "last_access_ts": d.last_access_ts,
                    "access_count": d.access_count,
                    "personalness": d.personalness,
                    "globalness": d.globalness,
                },
            )
            for d, vec in zip(docs_src, vecs_src, strict=False)
        ]
        qclient.upsert(collection_name=self._qdrant_collection, points=points)
        for d in docs_src:
            self._qdrant_access_counts[d.id] = d.access_count or 0
        self._use_qdrant = True
        self.index.reset()
        self.docs = []
        self.vectors = np.empty((0, self.dim), dtype=np.float32)
        self.doc_map.clear()
        self.vector_map.clear()

    def add(self, docs: list[MemoryDoc], vectors: ndarray) -> None:
        """Add ``docs`` and ``vectors`` to the store."""
        if vectors.shape[0] != len(docs):
            msg = "vectors/doc count mismatch"
            raise ValueError(msg)
        now = time.time()
        self._init_docs(docs, now)
        vectors = self._handle_vectors(docs, vectors)
        self._offload_to_qdrant(docs, vectors)

    def mark_access(self, ids: list[str], ts: float | None = None) -> None:
        if not ids:
            return
        ts = ts or time.time()
        for doc_id in set(ids):
            if doc_id in self.doc_map:
                d = self.doc_map[doc_id]
                d.last_access_ts = ts
                d.access_count = (d.access_count or 0) + 1
            elif self._use_qdrant and self._qdrant_client is not None:
                try:
                    prev_count = self._qdrant_access_counts.get(doc_id, 0)
                    new_count = prev_count + 1
                    self._qdrant_client.set_payload(
                        collection_name=self._qdrant_collection,
                        payload={"last_access_ts": ts, "access_count": new_count},
                        points=[doc_id],
                    )
                    self._qdrant_access_counts[doc_id] = new_count
                except Exception as exc:
                    logger.debug("Qdrant payload update failed for %s: %s", doc_id, exc)

    def search(self, query_vec: ndarray, k: int) -> tuple[list[MemoryDoc], ndarray]:
        if self._use_qdrant and self._qdrant_client is not None:
            results = self._qdrant_client.search(
                collection_name=self._qdrant_collection,
                query_vector=query_vec.tolist(),
                limit=k,
                with_payload=True,
                with_vectors=True,
            )
            docs: list[MemoryDoc] = []
            vecs: list[ndarray] = []
            for r in results:
                doc = self.doc_map.get(str(r.id))
                if doc is None:
                    payload = r.payload or {}
                    try:
                        channel = Channel(payload.get("channel", Channel.PERSONAL.value))
                    except Exception:
                        channel = Channel.PERSONAL
                    doc = MemoryDoc(
                        id=str(r.id),
                        text=payload.get("text", ""),
                        channel=channel,
                        metadata=payload.get("metadata", {}),
                        created_at_ts=payload.get("created_at_ts"),
                        last_access_ts=payload.get("last_access_ts"),
                        access_count=payload.get("access_count"),
                        personalness=payload.get("personalness", 1.0),
                        globalness=payload.get("globalness", 0.0),
                    )
                docs.append(doc)
                if getattr(r, "vector", None) is not None:
                    vecs.append(np.asarray(r.vector, dtype=np.float32))
                elif doc.id in self.vector_map:
                    vecs.append(self.vector_map[doc.id])
            return docs, np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)

        if self.index.ntotal == 0:
            return [], np.zeros((0, self.dim), dtype=np.float32)
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, idxs = self.index.search(q, k)
        docs = []
        vecs_list: list[ndarray] = []
        for idx in idxs[0]:
            if idx < 0 or idx >= len(self.docs):
                continue
            d = self.docs[idx]
            docs.append(d)
            vecs_list.append(self.vectors[idx])
        return (
            docs,
            np.vstack(vecs_list) if vecs_list else np.zeros((0, self.dim), dtype=np.float32),
        )


@dataclass
class CompositeWeights:
    """Weights used for composite scoring."""

    alpha_sim: float = 1.0
    beta_rerank: float = 2.0
    gamma_recency: float = 0.15
    eta_frequency: float = 0.10
    delta_globalness: float = 0.5
    lambda_personal_penalty: float = 1.0
    recency_tau_seconds: float = 30 * 24 * 3600


@dataclass
class RouterDecision:
    """Decision returned by :class:`Router`."""

    use_retrieval: bool
    target_channels: list[Channel]
    is_global_query: bool
    reason: str


class Router:
    """
    Naive heuristics to select memory channels.

    This can later be replaced with an LLM-based router. For now we use simple
    rules so the pipeline works without additional models.
    """

    PERSONAL_TRIGGERS = (
        "my ",
        "мой ",
        "моя ",
        "мои ",
        "меня ",
        "remember",
        "напомни",
    )
    GLOBAL_TRIGGERS = (
        "what is",
        "кто такой",
        "что такое",
        "define ",
        "how to",
        "почему",
        "error",
        "python",
    )

    def decide(self, query: str, *, project_context: bool = False) -> RouterDecision:
        """Return routing decision for ``query``."""

        q = query.strip().lower()
        if q.startswith(("no memory:", "nomem:")):
            return RouterDecision(
                use_retrieval=False,
                target_channels=[],
                is_global_query=True,
                reason="User opted out of retrieval",
            )
        if len(q) < MIN_QUERY_LEN:
            return RouterDecision(
                use_retrieval=False,
                target_channels=[],
                is_global_query=True,
                reason="Very short query",
            )

        has_personal = any(t in q for t in self.PERSONAL_TRIGGERS)
        has_global = any(t in q for t in self.GLOBAL_TRIGGERS)
        if has_personal and has_global:
            # Improvement: include both personal and global channels if cues for both are present
            channels: list[Channel] = [Channel.PERSONAL, Channel.GLOBAL]
            if project_context:
                channels.append(Channel.PROJECT)
            return RouterDecision(
                use_retrieval=True,
                target_channels=channels,
                is_global_query=False,
                reason="Personal and global cues present",
            )
        if has_personal and not has_global:
            channels = [Channel.PERSONAL]
            if project_context:
                channels.append(Channel.PROJECT)
            return RouterDecision(
                use_retrieval=True,
                target_channels=channels,
                is_global_query=False,
                reason="Personal cues dominate",
            )

        targets: list[Channel] = [Channel.GLOBAL]
        if project_context:
            targets.append(Channel.PROJECT)
        return RouterDecision(
            use_retrieval=True,
            target_channels=targets,
            is_global_query=True,
            reason="Default global/project retrieval",
        )


class SimpleRouter(Router):
    """Backward compatible alias for :class:`Router`."""


class TextEmbedder:
    """Bi‑encoder embeddings with graceful fallback."""

    def __init__(self, model_name: str = "bge-m3") -> None:
        self.model_name = model_name
        self.model: Any | None = None
        if ST is not None:
            try:
                self.model = ST(model_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load SentenceTransformer: %s", exc)

    @property
    def dim(self) -> int:
        if self.model is not None:
            return int(self.model.get_sentence_embedding_dimension())
        return 256

    def encode(self, texts: list[str]) -> ndarray:
        if self.model is not None:
            vecs = self.model.encode(texts)
            return np.asarray(vecs, dtype=np.float32)
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hash(t)
            rng = np.random.default_rng(abs(h) % (2**32))
            arr[i] = rng.normal(size=(self.dim,), loc=0.0, scale=1.0)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + EPSILON
        return arr / norms


class CrossEncoderReranker:
    """Cross‑encoder reranker with cosine fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model: Any | None = None
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load CrossEncoder: %s", exc)

    def score(self, query: str, docs: list[MemoryDoc], *, batch_size: int = 32) -> list[float]:
        if not docs:
            return []
        if self.model is None:
            return [0.0 for _ in docs]
        pairs = [(query, d.text) for d in docs]
        scores = self.model.predict(pairs, batch_size=batch_size)
        return cast("list[float]", scores.tolist())


class MemoriaHTTPClient:
    """Tiny HTTP client that talks to the Memoria service."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeouts: HTTPTimeouts | None = None,
        retry: int = 0,
        privacy_mode: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        timeouts = timeouts or HTTPTimeouts()
        self.timeouts = timeouts
        # ``httpx`` is only required at runtime; in the test-suite a tiny stub
        # object is provided which exposes only ``Client``.  Access attributes
        # via ``getattr`` so the stub works without implementing the whole
        # API.  When the optional pieces are missing we simply instantiate the
        # client without timeouts which is sufficient for the tests and keeps
        # behaviour backwards compatible.

        timeout_obj: Any | None = None
        timeout_cls = getattr(httpx, "Timeout", None)
        if timeout_cls is not None:
            try:
                timeout_obj = timeout_cls(read=timeouts.read, connect=timeouts.connect)
            except Exception:  # pragma: no cover - misbehaving stub
                timeout_obj = None

        client_cls = getattr(httpx, "AsyncClient", getattr(httpx, "Client", None))
        if client_cls is None:  # pragma: no cover - httpx stub is missing
            msg = "httpx client unavailable"
            raise RuntimeError(msg)
        try:
            self.client = client_cls(timeout=timeout_obj) if timeout_obj else client_cls()
        except TypeError:  # pragma: no cover - stub without timeout support
            self.client = client_cls()

        self.retry = retry
        if privacy_mode is None:
            privacy_mode = get_settings().security.privacy_mode
        self.privacy_mode = privacy_mode

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _post(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        idempotent: bool = False,
    ) -> httpx.Response:
        headers = self._headers()
        if idempotent:
            headers["Idempotency-Key"] = uuid.uuid4().hex
        attempt = 0
        while True:
            try:
                return await self.client.post(
                    f"{self.base_url}{path}",
                    content=json.dumps(payload) if payload is not None else None,
                    headers=headers,
                )
            except asyncio.CancelledError:  # pragma: no cover - cancellation
                raise
            except Exception as exc:  # pragma: no cover - network errors
                if attempt >= self.retry:
                    raise
                backoff = 2**attempt + secrets.randbelow(1000) / 1000
                logger.warning(
                    "POST %s failed (attempt %d/%d): %s", path, attempt + 1, self.retry, exc
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:  # pragma: no cover - cancellation
                    raise
                attempt += 1

    async def search(
        self,
        *,
        channel: Channel,
        query_vec: list[float],
        k: int = 20,
        query_text: str | None = None,
    ) -> list[MemoryDoc]:
        payload = {"vector": query_vec, "k": k, "channel": channel.value}
        if query_text is not None:
            payload["query"] = query_text
            payload["retriever"] = "hybrid"
        resp = await self._post("/memory/search", payload)
        resp.raise_for_status()
        data = resp.json()
        docs: list[MemoryDoc] = []
        for item in data.get("results", []):
            md = item.get("metadata", {})
            docs.append(
                MemoryDoc(
                    id=item.get("id", ""),
                    text=item.get("text", ""),
                    channel=channel,
                    metadata=md,
                    created_at_ts=md.get("created_at_ts"),
                    last_access_ts=md.get("last_access_ts"),
                    access_count=md.get("access_count"),
                    personalness=float(
                        md.get(
                            "personalness",
                            1.0 if channel is Channel.PERSONAL else 0.0,
                        )
                    ),
                    globalness=float(
                        md.get(
                            "globalness",
                            1.0 if channel is Channel.GLOBAL else 0.0,
                        )
                    ),
                )
            )
        return docs

    async def mark_access(self, ids: list[str], ts: float | None = None) -> float:
        if not ids:
            return ts or time.time()
        ts = ts or time.time()
        if self.privacy_mode == "strict":
            return ts
        payload = {"ids": ids, "accessed_at": ts}
        try:
            await self._post("/memory/mark_access", payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("mark_access failed: %s", exc)
        return ts

    async def record_feedback(self, mem_id: str, success: bool) -> None:
        payload = {"id": mem_id, "success": success}
        try:
            await self._post("/memory/feedback", payload, idempotent=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("feedback failed: %s", exc)


# ----------------------------- utilities ----------------------------------
def cosine_sim_matrix(a: ndarray, b: ndarray) -> ndarray:
    """Return cosine similarity matrix for ``a`` and ``b``.

    Works with both real NumPy and the lightweight test stub.
    """

    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    def _row_norms(x):
        try:
            return np.linalg.norm(x, axis=1, keepdims=True)  # type: ignore[attr-defined]
        except Exception:
            # Fallback for numpy stub without linalg
            import math as _math

            return np.asarray([[float(_math.sqrt(sum(float(v) * float(v) for v in row)))] for row in x], dtype=np.float32)

    a_norm = _row_norms(a_arr)
    b_norm = _row_norms(b_arr)

    def _normalize(arr, norms):
        try:
            return arr / (norms + EPSILON)
        except Exception:
            out = []
            for row, n in zip(arr, norms, strict=False):
                if isinstance(n, list):
                    dn = float(n[0]) if n else 1.0
                elif hasattr(n, "__iter__"):
                    vals = list(n)
                    dn = float(vals[0]) if vals else 1.0
                else:
                    dn = float(n)
                if dn <= 0:
                    dn = 1.0
                out.append([float(v) / dn for v in row])
            return np.asarray(out, dtype=np.float32)

    a_unit = _normalize(a_arr, a_norm)
    b_unit = _normalize(b_arr, b_norm)
    # Matrix multiply; fall back to manual when the stub lacks @ support
    try:
        return a_unit @ b_unit.T  # type: ignore[operator]
    except Exception:
        res = []
        for ra in a_unit:
            row = []
            for rb in b_unit:
                row.append(float(np.dot(ra, rb)))
            res.append(row)
        return np.asarray(res, dtype=np.float32)


def mmr_select(
    query_vec: ndarray,
    doc_vecs: ndarray,
    k: int,
    *,
    lambda_diversity: float = 0.7,
) -> list[int]:
    """Select ``k`` document indices using maximal marginal relevance."""

    docs = np.asarray(doc_vecs, dtype=np.float32)
    if len(docs) == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    sim_row = cosine_sim_matrix(q, docs)[0]
    # Convert to plain Python list for compatibility with numpy stub
    sim_to_query = sim_row.tolist() if hasattr(sim_row, "tolist") else list(sim_row)
    selected: list[int] = []
    candidates = set(range(len(docs)))
    while candidates and len(selected) < k:
        if not selected:
            idx = max(candidates, key=lambda ii: float(sim_to_query[ii]))
            selected.append(idx)
            candidates.remove(idx)
            continue
        # Compute redundancy for each candidate w.r.t. already selected set
        best_local = None
        best_gain = -1e18
        for ci in list(candidates):
            # Max similarity to any selected doc
            max_sim = 0.0
            for sj in selected:
                s = cosine_sim_matrix(docs[ci:ci + 1], docs[sj:sj + 1])[0][0]
                s_val = float(s if not hasattr(s, "tolist") else s.tolist())
                if s_val > max_sim:
                    max_sim = s_val
            gain = (1 - lambda_diversity) * float(sim_to_query[ci]) - lambda_diversity * max_sim
            if gain > best_gain:
                best_gain = gain
                best_local = ci
        selected.append(best_local)
        candidates.remove(best_local)
    return selected


class CompositeScorer:
    """Combine multiple signals into a single document score."""

    def __init__(self, weights: CompositeWeights) -> None:
        self.w = weights

    def _recency_boost(self, ts: float | None, *, now_ts: float | None = None) -> float:
        if ts is None:
            return 0.0
        ref = now_ts if now_ts is not None else time.time()
        age = max(0.0, ref - ts)
        return math.exp(-age / max(1.0, self.w.recency_tau_seconds))

    def _frequency_boost(self, cnt: int | None) -> float:
        if not cnt or cnt <= 0:
            return 0.0
        return math.log1p(cnt)

    def score(
        self,
        *,
        docs: list[MemoryDoc],
        sim: ndarray,
        rerank_scores: list[float] | None = None,
        is_global_query: bool = True,
    ) -> None:
        rer = rerank_scores or [0.0] * len(docs)
        now_ts = time.time()
        for i, d in enumerate(docs):
            d.score_sim = float(sim[i])
            rec = self._recency_boost(d.last_access_ts, now_ts=now_ts)
            freq = self._frequency_boost(d.access_count)
            penalty = self.w.lambda_personal_penalty * (d.personalness if is_global_query else 0.0)
            bonus_global = self.w.delta_globalness * d.globalness
            d.score_rerank = float(rer[i])
            # Clamp extremely small similarities to reduce floating noise
            sim_adj = round(d.score_sim, 12)
            d.score_composite = (
                self.w.alpha_sim * sim_adj
                + self.w.beta_rerank * d.score_rerank
                + self.w.gamma_recency * rec
                + self.w.eta_frequency * freq
                + bonus_global
                - penalty
            )


@dataclass
class PipelineConfig:
    """
    Configuration options for :class:`RAGRouterPipeline`.

    The pipeline can run a Maximal Marginal Relevance (MMR) deduplication step
    before scoring. Set ``use_mmr`` to enable it and tune ``mmr_lambda`` to
    balance relevance vs. diversity (``0.0`` = relevance, ``1.0`` = diversity).
    """

    base_url: str
    api_key: str | None = None
    topk_per_channel: int = 32
    final_k: int = 8
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    use_cross_encoder: bool = False
    embedder_model: str = "bge-m3"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    weights: CompositeWeights = field(default_factory=CompositeWeights)
    router: Router = field(default_factory=SimpleRouter)
    personal_threshold: int = 10000
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "personal"


class RAGRouterPipeline:
    """High-level retrieval pipeline combining routing, search and scoring."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.router = config.router
        self.embedder = TextEmbedder(config.embedder_model)
        self.reranker = None
        if config.use_cross_encoder:
            rr = CrossEncoderReranker(config.cross_encoder_model)
            if getattr(rr, "model", None) is not None:
                self.reranker = rr
        self.client = MemoriaHTTPClient(config.base_url, config.api_key)
        self.scorer = CompositeScorer(config.weights)
        self.personal_store = LocalPersonalStore(
            self.embedder.dim,
            threshold=config.personal_threshold,
            qdrant_url=config.qdrant_url,
            collection=config.qdrant_collection,
            api_key=config.qdrant_api_key,
        )
        self._last_results: list[MemoryDoc] = []

        # Optional Postgres persistent store
        settings = get_settings()
        db_cfg = getattr(settings, "database", None)
        self.memory_store = None
        if (
            db_cfg
            and getattr(db_cfg, "engine", "sqlite") == "postgres"
            and getattr(db_cfg, "postgres_dsn", None)
        ):
            try:  # pragma: no cover - best effort
                from memory_system.core.postgres_store import PostgresMemoryStore

                self.memory_store = PostgresMemoryStore(db_cfg.postgres_dsn)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.memory_store.initialise())
                else:
                    loop.run_until_complete(self.memory_store.initialise())
            except Exception:  # pragma: no cover - optional dependency
                logger.debug("failed to initialise Postgres store", exc_info=True)
                self.memory_store = None

    def _lm_token_probs(self, prompt: str) -> Sequence[float]:
        """
        Return LM token probabilities for ``prompt``.

        This default implementation returns an empty list; callers may
        monkeypatch or subclass to provide model-specific behaviour.
        """

        return []

    def retrieve(
        self,
        query: str,
        *,
        project_context: bool = False,
        effort: Literal["low", "med", "high"] = "med",
        max_k: int | None = None,
        max_context_tokens: int | None = None,
        max_cross_rerank_n: int | None = None,
        timeout: float | None = None,
    ) -> tuple[list[MemoryDoc], RouterDecision]:
        from types import SimpleNamespace as _NS

        settings = get_settings()
        # Be defensive: provide sensible defaults when effort budgets are absent.
        _eff = getattr(settings, "effort", None)
        budgets = getattr(_eff, effort, None) if _eff is not None else None
        if budgets is None:
            budgets = _NS(
                max_k=32,
                max_context_tokens=1024,
                max_cross_rerank_n=16,
                timeout_seconds=2.0,
            )
        max_k = max_k or budgets.max_k
        max_context_tokens = max_context_tokens or budgets.max_context_tokens
        max_cross_rerank_n = max_cross_rerank_n or budgets.max_cross_rerank_n
        timeout = timeout or budgets.timeout_seconds
        start = time.time()
        threshold = getattr(getattr(settings, "rag", object()), "entropy_threshold", None)
        probs = self._lm_token_probs(query)
        if threshold is not None and probs:
            entropy = calc_entropy(probs)
            if entropy <= threshold:
                decision = RouterDecision(
                    use_retrieval=False,
                    target_channels=[],
                    is_global_query=True,
                    reason="Low entropy",
                )
                return [], decision

        decision = self.router.decide(query, project_context=project_context)
        if not decision.use_retrieval:
            return [], decision

        qvec = self.embedder.encode([query])[0]
        force_hybrid = _lang_alphabet_mismatch(query)
        all_docs: list[MemoryDoc] = []
        all_vecs: list[ndarray] = []
        remote_ids: list[str] = []
        per_channel_k = min(self.cfg.topk_per_channel, max_k)
        # Improvement: concurrent retrieval for remote channels and adaptive search
        remote_tasks = []
        remote_channels: list[Channel] = []
        for ch in decision.target_channels:
            if time.time() - start > timeout:
                logger.warning("retrieve timeout after %.2f s", timeout)
                break
            if ch is Channel.PERSONAL:
                docs, dvecs = self.personal_store.search(qvec, per_channel_k)
                if docs and self.cfg.use_mmr and dvecs.shape[0] > self.cfg.final_k:
                    keep_idx = mmr_select(
                        qvec,
                        dvecs,
                        k=min(self.cfg.final_k * 3, dvecs.shape[0]),
                        lambda_diversity=self.cfg.mmr_lambda,
                    )
                    docs = [docs[i] for i in keep_idx]
                    dvecs = dvecs[keep_idx]
                if docs:
                    all_docs.extend(docs)
                    all_vecs.append(dvecs)
            else:
                # Prepare remote search task for concurrent execution
                task = self.client.search(
                    channel=ch,
                    query_vec=qvec.tolist(),
                    k=per_channel_k,
                    query_text=query if force_hybrid else None,
                )
                remote_tasks.append(task)
                remote_channels.append(ch)
        # Execute all remote searches concurrently
        remote_results: list[Any] = []
        if remote_tasks:
            try:
                remote_results = asyncio.run(
                    cast(
                        "Coroutine[Any, Any, list[Any]]",
                        asyncio.gather(*remote_tasks, return_exceptions=True),
                    )
                )
            except Exception as exc:
                logger.warning("Concurrent remote search failed: %s", exc)
                remote_results = []
        # Process remote search results
        all_remote_docs: list[MemoryDoc] = []
        doc_counts: list[int] = []
        for idx, result in enumerate(remote_results):
            ch = remote_channels[idx]
            if isinstance(result, Exception):
                logger.warning("Search on channel %s failed: %s", ch, result)
                continue
            docs = result
            if not docs:
                continue
            all_remote_docs.extend(docs)
            doc_counts.append(len(docs))
            # Track remote IDs for marking access
            remote_ids.extend([d.id for d in docs])
        # Encode all remote documents' text in one batch for efficiency
        if all_remote_docs:
            remote_vecs = self.embedder.encode([d.text for d in all_remote_docs])
            offset = 0
            for count in doc_counts:
                if count == 0:
                    continue
                docs_slice = all_remote_docs[offset : offset + count]
                vecs_slice = remote_vecs[offset : offset + count]
                offset += count
                if self.cfg.use_mmr and vecs_slice.shape[0] > self.cfg.final_k:
                    keep_idx = mmr_select(
                        qvec,
                        vecs_slice,
                        k=min(self.cfg.final_k * 3, vecs_slice.shape[0]),
                        lambda_diversity=self.cfg.mmr_lambda,
                    )
                    docs_slice = [docs_slice[i] for i in keep_idx]
                    vecs_slice = vecs_slice[keep_idx]
                all_docs.extend(docs_slice)
                all_vecs.append(vecs_slice)

        if not all_docs:
            # Improvement: adaptive fallback search in other channels if no results
            alt_channels: list[Channel] = []
            if Channel.GLOBAL not in decision.target_channels:
                alt_channels.append(Channel.GLOBAL)
            if Channel.PERSONAL not in decision.target_channels:
                alt_channels.append(Channel.PERSONAL)
            if alt_channels:
                logger.debug(
                    "No retrieval results from channels %s, trying fallback: %s",
                    decision.target_channels,
                    alt_channels,
                )
                for ch in alt_channels:
                    if time.time() - start > timeout:
                        logger.warning("retrieve timeout during fallback after %.2f s", timeout)
                        break
                    if ch is Channel.PERSONAL:
                        docs, dvecs = self.personal_store.search(qvec, per_channel_k)
                    else:
                        try:
                            docs = asyncio.run(
                                self.client.search(
                                    channel=ch,
                                    query_vec=qvec.tolist(),
                                    k=per_channel_k,
                                    query_text=query if force_hybrid else None,
                                )
                            )
                        except Exception as exc:
                            logger.warning("Fallback search on channel %s failed: %s", ch, exc)
                            docs = []
                        if docs:
                            dvecs = self.embedder.encode([d.text for d in docs])
                    if not docs:
                        continue
                    if ch is not Channel.PERSONAL:
                        remote_ids.extend([d.id for d in docs])
                    if self.cfg.use_mmr and dvecs.shape[0] > self.cfg.final_k:
                        keep_idx = mmr_select(
                            qvec,
                            dvecs,
                            k=min(self.cfg.final_k * 3, dvecs.shape[0]),
                            lambda_diversity=self.cfg.mmr_lambda,
                        )
                        docs = [docs[i] for i in keep_idx]
                        dvecs = dvecs[keep_idx]
                    all_docs.extend(docs)
                    all_vecs.append(dvecs)
                if not all_docs:
                    return [], decision
            else:
                return [], decision

        doc_vecs = np.vstack(all_vecs)
        sim = cosine_sim_matrix(qvec[None, :], doc_vecs)[0]

        if self.cfg.use_mmr and len(all_docs) > self.cfg.final_k:
            try:
                items = list(zip(range(len(all_docs)), doc_vecs.tolist(), strict=False))
                order = mmr(items, qvec.tolist(), lambda_=self.cfg.mmr_lambda)
                max_cand = min(len(order), self.cfg.final_k * 3)
                keep_idx = order[:max_cand]
            except Exception as exc:  # pragma: no cover
                logger.warning("MMR failed, falling back to top-K: %s", exc)
                keep_idx = np.argsort(-sim)[: min(len(sim), self.cfg.final_k * 3)]
            all_docs = [all_docs[i] for i in keep_idx]
            doc_vecs = doc_vecs[keep_idx]
            sim = sim[keep_idx]

        if sim.size:
            top_n = min(len(sim), 64)
            top_vals = np.sort(sim)[-top_n:]
            sim_min = float(top_vals.min())
            sim_max = float(top_vals.max())
            sim_norm = (sim - sim_min) / (sim_max - sim_min + EPSILON)
        else:
            sim_norm = sim

        bm = np.array(
            [float(d.metadata.get("bm25_score", 0.0)) for d in all_docs],
            dtype=float,
        )
        if bm.size and np.any(bm):
            top_bm = np.sort(bm)[-top_n:]
            bm_min = float(top_bm.min())
            bm_max = float(top_bm.max())
            if bm_max - bm_min > EPSILON:
                bm_norm = (bm - bm_min) / (bm_max - bm_min + EPSILON)
            else:
                bm_norm = np.zeros_like(bm)
            for i, d in enumerate(all_docs):
                d.metadata["bm25_score"] = float(bm_norm[i])
        else:
            bm_norm = np.zeros_like(bm)

        sim = (sim_norm + bm_norm) / 2.0 if bm_norm.any() else sim_norm

        rerank_scores: list[float] | None = None
        if self.reranker is not None:
            top_n = min(len(all_docs), max_cross_rerank_n)
            top_idx = np.argsort(-sim)[:top_n]
            cand_docs = [all_docs[i] for i in top_idx]
            rer = np.array(self.reranker.score(query, cand_docs), dtype=float)
            if rer.size:
                rer = (rer - rer.min()) / (rer.max() - rer.min() + EPSILON)
            rerank_scores = [0.0] * len(all_docs)
            for j, i_orig in enumerate(top_idx):
                rerank_scores[i_orig] = float(rer[j])

        self.scorer.score(
            docs=all_docs,
            sim=sim,
            rerank_scores=rerank_scores,
            is_global_query=decision.is_global_query,
        )
        final = sorted(all_docs, key=lambda d: d.score_composite, reverse=True)[
            : min(self.cfg.final_k, max_k)
        ]
        final = self._bandit_rerank(final)
        trimmed: list[MemoryDoc] = []
        total = 0
        for d in final:
            tok = len(d.text.split())
            if total + tok > max_context_tokens:
                logger.warning(
                    "context token limit exceeded: %d > %d", total + tok, max_context_tokens
                )
                break
            total += tok
            trimmed.append(d)
        final = trimmed
        self._last_results = final
        ts = time.time()
        remote_mark = [d.id for d in final if d.id in remote_ids]
        if remote_mark and self.client.privacy_mode != "strict":
            try:
                asyncio.run(self.client.mark_access(remote_mark, ts=ts))
            except Exception as exc:  # pragma: no cover
                logger.warning("mark_access failed: %s", exc)
        personal_mark = [d.id for d in final if d.channel is Channel.PERSONAL]
        if personal_mark:
            self.personal_store.mark_access(personal_mark, ts=ts)
        return final, decision

    def add_personal_memories(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[dict[str, Any] | None] | None = None,
    ) -> None:
        """Index personal memories in the local store."""
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        docs = [
            MemoryDoc(
                id=ids[i],
                text=texts[i],
                channel=Channel.PERSONAL,
                metadata=metadatas[i] or {},
            )
            for i in range(len(texts))
        ]
        vecs = self.embedder.encode(texts)
        self.personal_store.add(docs, vecs)

    # ---------------------------- feedback API ---------------------------
    def _bandit_rerank(self, docs: list[MemoryDoc]) -> list[MemoryDoc]:
        if not docs:
            return docs
        try:

            class _Stub:
                def __init__(self, d: MemoryDoc) -> None:
                    self.memory_id = d.id
                    self.metadata = d.metadata

            stubs = [_Stub(d) for d in docs]
            ranked = rerank_with_bandit(cast("Sequence[Any]", stubs))
            id_map = {d.id: d for d in docs}
            return [id_map[s.memory_id] for s in ranked]
        except Exception:  # pragma: no cover - bandit optional
            return docs

    def _apply_feedback(self, doc: MemoryDoc, success: bool) -> None:
        meta = dict(doc.metadata or {})
        trials = int(meta.get("trial_count", 0)) + 1
        successes = int(meta.get("success_count", 0)) + (1 if success else 0)
        meta.update({"trial_count": trials, "success_count": successes})
        doc.metadata = meta
        if doc.channel is Channel.PERSONAL:
            for d in self.personal_store.docs:
                if d.id == doc.id:
                    d.metadata = meta
                    break
        else:
            try:
                asyncio.run(self.client.record_feedback(doc.id, success))
            except Exception as exc:  # pragma: no cover
                logger.warning("record_feedback failed: %s", exc)
        self._last_results = self._bandit_rerank(self._last_results)

    def mark_success(self, mem: MemoryDoc | str) -> None:
        doc = self._resolve_doc(mem)
        if doc:
            self._apply_feedback(doc, True)

    def mark_failure(self, mem: MemoryDoc | str) -> None:
        doc = self._resolve_doc(mem)
        if doc:
            self._apply_feedback(doc, False)

    def _resolve_doc(self, mem: MemoryDoc | str) -> MemoryDoc | None:
        if isinstance(mem, MemoryDoc):
            return mem
        for d in self._last_results:
            if d.id == mem:
                return d
        for d in self.personal_store.docs:
            if d.id == mem:
                return d
        return None


def graph_search(query: str) -> list[MemoryDoc]:
    """
    Search the knowledge graph for cards related to ``query``.

    The function extracts naive triples from the query and tries to find paths
    between the detected entities inside the global ``graph_store``.  Cards
    attached to the nodes and edges along those paths are returned as
    :class:`MemoryDoc` instances.
    """

    results: list[MemoryDoc] = []
    seen: set[str] = set()
    for subj, _pred, obj in extract_triples(query):
        for cid in graph_store.find_cards_between(subj, obj):
            if cid in seen:
                continue
            seen.add(cid)
            text = graph_store.card_content.get(cid, "")
            results.append(MemoryDoc(id=cid, text=text, channel=Channel.PERSONAL))
    return results


__all__ = [
    "Channel",
    "CompositeScorer",
    "CompositeWeights",
    "CrossEncoderReranker",
    "LocalPersonalStore",
    "MemoriaHTTPClient",
    "MemoryDoc",
    "PipelineConfig",
    "RAGRouterPipeline",
    "Router",
    "RouterDecision",
    "SimpleRouter",
    "TextEmbedder",
    "cosine_sim_matrix",
    "graph_search",
    "mmr_select",
]
