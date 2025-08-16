"""
memory_system.unified_memory
================================
High‑level, **framework‑agnostic** helper functions that wrap the lower
level storage / vector components.  Nothing in here depends on FastAPI
or any other web framework – the goal is to let notebooks, background
jobs, or other services reuse the same persistence logic without pulling
in heavy HTTP deps.

Notes
-----
* All functions are **async**.  They accept an optional ``store`` kwarg –
  any object that implements ``add_memory``, ``search_memory``,
  ``delete_memory`` and ``update_metadata``.  If not supplied the helper
  tries to obtain the application‑scoped store via
  :pyfunc:`memory_system.core.store.get_memory_store`.
* Docstrings follow **PEP 257** and type hints are 100 % complete so that
  MyPy / Ruff‑strict pass cleanly.

"""

from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import hashlib
import logging
import math
import os
import random
import re
import secrets
import unicodedata
import uuid
from collections import defaultdict

# stdlib
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextvars import ContextVar, copy_context

# local
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast

from memory_system import sensory_buffer
from memory_system.utils.cache import SmartCache
from memory_system.utils.dependencies import require_httpx
from memory_system.utils.security import EnhancedPIIFilter

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from memory_system.core.store import Case

TTL_BUMP_PROB = 0.1
TTL_BUMP_FACTOR = 1.5

# Number of successful contribution updates required before a shadow memory
# graduates to a regular memory.
SHADOW_PROMOTION_THRESHOLD = 3.0

RRF_K = 60
WEIGHT_EPSILON = 1e-6
EPSILON = 1e-9

DEDUP_SECRET = os.getenv("DEDUP_KEY", "").encode()

CONTRADICTION_THRESHOLD = 3
STALE_TTL_SECONDS = 30 * 24 * 3600
SOURCE_CONTRADICTIONS: defaultdict[str, int] = defaultdict(int)


@dataclass(slots=True)
class ExperienceCard:
    """Structured description of a past experience."""

    situation: str | None = None
    approach: str | None = None
    tools: str | None = None
    result: str | None = None
    lesson: str | None = None
    antipattern: str | None = None
    verifiability: str | None = None
    source_hash: str | None = None
    lang: str | None = None
    summary: str | None = None
    summary_en: str | None = None
    success_count: int = 0
    trial_count: int = 0
    delta_i: float = 1.0


@dataclass(slots=True)
class PersonalCard:
    """Light‑weight card describing promoted personal knowledge."""

    title: str
    claim: str
    sources: list[str]
    license: str
    trust_score: float = 0.0
    lang: str | None = None
    summary: str | None = None
    summary_en: str | None = None
    success_count: int = 0
    trial_count: int = 0
    delta_i: float = 1.0


@dataclass(slots=True)
class CodeVerifier:
    """Verification details for code memories."""

    test_suite_path: str
    build_hash: str


@dataclass(slots=True)
class MathVerifier:
    """Verification details for mathematical memories."""

    check: str


@dataclass(slots=True)
class Memory:
    """Simple memory record used by helper functions."""

    memory_id: str
    text: str
    created_at: _dt.datetime
    valid_from: _dt.datetime | None = None
    valid_to: _dt.datetime | None = None
    tx_from: _dt.datetime | None = None
    tx_to: _dt.datetime | None = None
    valence: float = 0.0
    emotional_intensity: float = 0.0
    importance: float = 0.0
    episode_id: str | None = None
    modality: str = "text"
    connections: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None
    schema_type: str | None = None
    card: ExperienceCard | None = None
    memory_type: Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] = (
        "episodic"
    )
    pinned: bool = False
    ttl_seconds: int | None = None
    last_used: _dt.datetime | None = None
    success_score: float | None = None
    decay: float | None = None
    verifier: CodeVerifier | MathVerifier | None = None
    _tokens: set[str] | None = field(default=None, init=False, repr=False, compare=False)
    _embedding: Any | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.valid_from is None:
            self.valid_from = self.created_at
        if self.valid_to is None:
            self.valid_to = _dt.datetime.max.replace(tzinfo=_dt.UTC)
        if self.tx_from is None:
            self.tx_from = self.created_at
        if self.tx_to is None:
            self.tx_to = _dt.datetime.max.replace(tzinfo=_dt.UTC)


class SearchResults(list[Memory]):
    """
    Container returned by :func:`search`.

    It behaves like a regular ``list`` of :class:`Memory` objects while also
    exposing the subset of shadow memories via the ``shadow`` attribute.
    Callers constructing final answers should ignore ``shadow`` memories.
    """

    #: memories tagged as ``shadow`` surfaced during retrieval
    shadow: list[Memory]

    def __init__(self, primary: Sequence[Memory], shadow: Sequence[Memory]):
        super().__init__(list(primary) + list(shadow))
        self.shadow = list(shadow)


class MemoryStoreProtocol(Protocol):
    """Minimal protocol expected from a memory store."""

    async def add_memory(self, memory: Memory) -> None: ...

    async def add_many(self, memories: Sequence[Memory], *, batch_size: int = 100) -> None: ...

    async def add_memories_streaming(
        self,
        iterator: Iterable[dict[str, Any]] | AsyncIterator[dict[str, Any]],
        *,
        batch_size: int = 100,
        save_interval: int | None = None,
    ) -> int: ...

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter: MutableMapping[str, Any] | None = None,
        level: int | None = None,
        context: MutableMapping[str, Any] | None = None,
    ) -> SearchResults: ...

    async def get(self, memory_id: str) -> Memory | None: ...

    async def search_cases(self, problem: str, k: int = 5) -> list[Case]: ...

    async def delete_memory(self, memory_id: str) -> None: ...

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: MutableMapping[str, Any] | None = None,
        importance: float | None = None,
        importance_delta: float | None = None,
        valence: float | None = None,
        valence_delta: float | None = None,
        emotional_intensity: float | None = None,
        emotional_intensity_delta: float | None = None,
        memory_type: (
            Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] | None
        ) = None,
        pinned: bool | None = None,
        ttl_seconds: int | None = None,
        last_used: _dt.datetime | None = None,
        success_score: float | None = None,
        decay: float | None = None,
    ) -> Memory:
        """
        Update fields or scores of a memory.

        Absolute parameters overwrite stored values while ``*_delta`` fields
        apply relative adjustments.
        """
        ...

    async def list_recent(
        self,
        *,
        n: int = 20,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
    ) -> list[Memory]: ...

    async def upsert_scores(self, scores: Sequence[tuple[str, float]]) -> None: ...

    async def top_n_by_score(
        self,
        n: int,
        *,
        level: int | None = None,
        metadata_filter: MutableMapping[str, Any] | None = None,
        weights: ListBestWeights | None = None,
        ids: Sequence[str] | None = None,
    ) -> Sequence[Memory]: ...


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


_feedback_heap: list[Memory] = []


def _get_feedback_cfg() -> SimpleNamespace:
    """Return feedback sampling configuration or defaults."""
    try:  # lazy import to avoid heavy deps at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        fb = getattr(cfg, "feedback", None)
        if fb is None:
            raise AttributeError
        return cast("SimpleNamespace", fb)
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return SimpleNamespace(alpha=1.0)


def sample_feedback(k: int = 1) -> list[Memory]:
    """Return up to ``k`` memories sampled by quality error."""
    if not _feedback_heap or k <= 0:
        return []

    cfg = _get_feedback_cfg()
    items = list(_feedback_heap)
    weights: list[float] = []
    for m in items:
        meta = m.metadata or {}
        delta = float(meta.get("delta_i", 0.0))
        weights.append((abs(delta) + WEIGHT_EPSILON) ** cfg.alpha)
    chosen = random.choices(items, weights=weights, k=min(k, len(items)))
    for m in chosen:
        try:
            _feedback_heap.remove(m)
        except ValueError:  # pragma: no cover - defensive
            continue
    return chosen


DEFAULT_ASYNC_TIMEOUT = 5  # seconds – safety net for accidental long‑running operations.
_ASYNC_TIMEOUT: float | None = None
RECENCY_TAU = 86_400.0  # seconds – time constant for recency/EMA updates.

# Process-wide default store used when no explicit store is provided.
_DEFAULT_STORE: ContextVar[MemoryStoreProtocol | None] = ContextVar("_DEFAULT_STORE", default=None)

_CACHE: SmartCache | None = None
_FPRINT_CACHE: SmartCache | None = None
_BG_TASKS: set[asyncio.Task[Any]] = set()


def _get_async_timeout() -> float:
    """Return async timeout from settings or default."""
    global _ASYNC_TIMEOUT
    if _ASYNC_TIMEOUT is None:
        try:  # local import to avoid heavy config dependency at import time
            from memory_system.settings import get_settings

            _ASYNC_TIMEOUT = float(get_settings().performance.async_timeout)
        except (ImportError, AttributeError):  # pragma: no cover - settings module optional
            _ASYNC_TIMEOUT = float(DEFAULT_ASYNC_TIMEOUT)
    return _ASYNC_TIMEOUT


T = TypeVar("T")


async def _wait_with_timeout(awaitable: Awaitable[T], desc: str) -> T:
    """Run ``awaitable`` with timeout and log when exceeded."""
    timeout = _get_async_timeout()
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        logger.warning("%s timed out after %.1f seconds", desc, timeout)
        raise


def _get_cache() -> SmartCache:
    """Return module-wide cache instance configured from settings."""
    global _CACHE
    if _CACHE is None:
        try:  # local import to avoid heavy config dependency at import time
            from memory_system.settings import get_settings

            cfg = get_settings()
            _CACHE = SmartCache(max_size=cfg.cache.size, ttl=cfg.cache.ttl_seconds)
        except (ImportError, AttributeError):  # pragma: no cover - settings module optional
            _CACHE = SmartCache()
    return _CACHE


def _get_fp_cache() -> SmartCache:
    """Return cache for fast fingerprint-based deduplication."""
    global _FPRINT_CACHE
    if _FPRINT_CACHE is None:
        try:  # defer heavy settings import
            from memory_system.settings import get_settings

            cfg = get_settings().performance
            _FPRINT_CACHE = SmartCache(max_size=cfg.cache_size, ttl=cfg.cache_ttl_seconds)
        except (ImportError, AttributeError):  # pragma: no cover - settings optional
            _FPRINT_CACHE = SmartCache()
    return _FPRINT_CACHE


def _spawn_bg(coro: Coroutine[Any, Any, Any], name: str = "bg") -> asyncio.Task[Any]:
    """Launch *coro* as a background task and track it for shutdown."""
    ctx = copy_context()
    task: asyncio.Task[Any] = asyncio.create_task(coro, name=name, context=ctx)
    _BG_TASKS.add(task)

    def _done(t: asyncio.Task[Any]) -> None:
        _BG_TASKS.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.exception("bg task failed", exc_info=exc)

    task.add_done_callback(_done)
    return task


async def close() -> None:
    """Cancel any outstanding background tasks."""
    for t in list(_BG_TASKS):
        t.cancel()
    await asyncio.gather(*_BG_TASKS, return_exceptions=True)


def set_default_store(store: MemoryStoreProtocol) -> None:
    """Register *store* as the fallback for all helper functions."""
    _DEFAULT_STORE.set(store)


def get_default_store() -> MemoryStoreProtocol | None:
    """Return the currently registered default store, if any."""
    return _DEFAULT_STORE.get()


async def _resolve_store(
    store: MemoryStoreProtocol | None = None,
) -> MemoryStoreProtocol:
    """
    Return a concrete store instance.

    The rules are:
    1. If *store* is given → use it.
    2. Else return the process‑wide default registered via :func:`set_default_store`.
    """
    if store is not None:
        return store

    resolved = get_default_store()
    if resolved is None:
        raise RuntimeError("Memory store has not been initialised.")
    return resolved


def _get_dynamics() -> SimpleNamespace:
    """Return dynamics configuration or sane defaults."""
    try:  # lazy import to avoid heavy deps at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        dyn = getattr(cfg, "dynamics", None)
        if dyn is None:
            raise AttributeError
        return cast("SimpleNamespace", dyn)
    except (ImportError, AttributeError):  # pragma: no cover - settings module optional
        return SimpleNamespace(
            initial_intensity=0.0,
            reinforce_delta=0.1,
            decay_rate=30.0,
            decay_law="exponential",
        )


# Registry for optional verifier hooks used during the pre-write pipeline.
_VERIFIERS: dict[str, list[Callable[[str], bool]]] = {}


def register_verifier(memory_type: str, func: Callable[[str], bool]) -> None:
    """Register a verification *func* for ``memory_type``."""
    _VERIFIERS.setdefault(memory_type, []).append(func)


def _run_verifiers(text: str, memory_type: str) -> None:
    """Run registered verifiers for ``memory_type`` raising on failure."""
    for func in _VERIFIERS.get(memory_type, []):
        try:
            ok = func(text)
        except Exception as exc:  # pragma: no cover - verifier bug
            logger.warning("verifier %s failed: %s", func, exc)
            ok = False
        if not ok:
            raise ValueError("verification failed")


def _get_min_score() -> float:
    try:  # lazy import to avoid heavy dependency at import time
        from memory_system.settings import get_settings

        return float(get_settings().ranking.min_score)
    except Exception:  # pragma: no cover - optional settings
        return 0.0


def _cross_encoder_score(text: str) -> float:
    """Score *text* using the optional cross encoder."""
    try:
        from memory_system.core.cross_reranker import _load_model

        model = _load_model()
        if model is None:
            return 1.0
        return float(model.predict([(text, text)])[0])
    except Exception:  # pragma: no cover - model optional/broken
        logger.warning("cross-encoder scoring failed", exc_info=True)
        return 0.0


def _canon(s: str) -> str:
    """Return canonical form: NFKC, strip punctuation, collapse spaces, lower."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split()).lower()
    return s


def _text_fingerprint(
    text: str,
    *,
    role: str = "",
    modality: str = "text",
    lang: str = "",
) -> str:
    """Return stable fingerprint of normalized text and context."""
    msg = f"{role}|{modality}|{lang}|{_canon(text)}".encode()
    if DEDUP_SECRET:
        return hashlib.sha256(DEDUP_SECRET + b"\x00" + msg).hexdigest()
    return hashlib.blake2b(msg, digest_size=16).hexdigest()


class DuplicateMemoryError(Exception):
    """Raised when an incoming memory is a near-duplicate."""


def _deep_dedup_enabled() -> bool:
    """Return whether background deep deduplication is enabled."""
    try:  # lazy import to avoid heavy settings dependency
        from memory_system.settings import get_settings

        return bool(get_settings().performance.deep_dedup)
    except Exception:  # pragma: no cover - settings optional
        return True


async def _async_deep_dedup(text: str, store: MemoryStoreProtocol) -> None:
    """Run near-duplicate detection in the background using the cross encoder."""
    from memory_system.core import dedup
    from memory_system.core.cross_reranker import order_by_cross_encoder

    try:
        candidates = await _wait_with_timeout(
            store.search_memory(query=text, k=5),
            "store.search_memory",
        )
    except Exception:  # pragma: no cover - search failure
        return
    if not candidates:
        return
    from memory_system.core.store import Memory as _StoreMem

    ordered = order_by_cross_encoder(cast("Sequence[_StoreMem]", list(candidates)), text)
    if dedup.is_near_duplicate(text, [m.text for m in ordered]):
        logger.info("background dedup found near-duplicate for incoming memory")


async def _pre_write_pipeline(
    text: str,
    memory_type: str,
    store: MemoryStoreProtocol,
    *,
    role: str = "",
    modality: str = "text",
    lang: str = "",
) -> None:
    """Run pre-write checks before persisting a memory."""
    # Fast fingerprint-based deduplication
    cache = _get_fp_cache()
    fingerprint = _text_fingerprint(text, role=role, modality=modality, lang=lang)
    if cache.get(fingerprint):
        logger.info("drop dup fp=%s", fingerprint[:8])
        raise DuplicateMemoryError("near-duplicate")
    cache.put(fingerprint, True)

    # Optional deep deduplication via background cross-encoder check
    if _deep_dedup_enabled():
        _spawn_bg(_async_deep_dedup(text, store), "deep_dedup")

    # Type-aware verifiers
    _run_verifiers(text, memory_type)

    # Cross-encoder threshold
    score = _cross_encoder_score(text)
    threshold = _get_min_score()
    if score < threshold:
        logger.info("dropping memory due to low score %.3f < %.3f", score, threshold)
        raise ValueError("draft score below threshold")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def add(
    text: str,
    *,
    valence: float = 0.0,
    emotional_intensity: float | None = None,
    importance: float = 0.0,
    episode_id: str | None = None,
    modality: str = "text",
    connections: MutableMapping[str, float] | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    verifier: CodeVerifier | MathVerifier | MutableMapping[str, Any] | None = None,
    schema_type: str | None = None,
    card: ExperienceCard | MutableMapping[str, Any] | None = None,
    memory_type: Literal[
        "sensory", "working", "episodic", "semantic", "skill", "lesson"
    ] = "episodic",
    pinned: bool = False,
    ttl_seconds: int | None = None,
    last_used: _dt.datetime | None = None,
    success_score: float | None = None,
    decay: float | None = None,
    context: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Persist a *text* record with optional *metadata* and return a **Memory**.

    Args:
        text (str): Raw textual content of the memory.
        valence (float, optional): Emotional valence. Defaults to 0.0.
        emotional_intensity (float, optional): Intensity of emotion. Defaults to 0.0.
        importance (float, optional): Importance score. Defaults to 0.0.
        episode_id (str | None, optional): Episode identifier. Defaults to None.
        modality (str, optional): Modality type. Defaults to "text".
        connections (MutableMapping[str, float] | None, optional): Connections. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): Arbitrary metadata. Defaults to None.
        context (MutableMapping[str, Any] | None, optional): Contextual info influencing ranking weights.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The persisted memory object.

    """
    if not text:
        raise ValueError("text must not be empty")

    st = await _resolve_store(store)
    meta: dict[str, Any] = dict(copy.deepcopy(metadata) if metadata else {})
    await _pre_write_pipeline(
        text,
        memory_type,
        st,
        role=str(meta.get("role", "")),
        modality=modality,
        lang=str(meta.get("lang", "")),
    )

    now = _dt.datetime.now(_dt.UTC)
    if verifier is not None:
        if isinstance(verifier, CodeVerifier | MathVerifier):
            meta.setdefault("verifier", asdict(verifier))
            verifier_obj = verifier
        else:
            meta.setdefault("verifier", dict(verifier))
            verifier_obj = None
    else:
        verifier_obj = None
    meta.setdefault("trust_score", 1.0)
    meta.setdefault("error_flag", False)
    if schema_type is not None:
        meta.setdefault("schema_type", schema_type)
    card_obj: ExperienceCard | None = None
    if card is not None:
        if isinstance(card, ExperienceCard):
            card_obj = card
            card_dict = asdict(card)
        else:
            card_dict = dict(card)
            card_obj = ExperienceCard(**card_dict)
        delta = card_dict.get("delta_i", meta.get("delta_i", 1.0))
        card_dict.setdefault("delta_i", delta)
        meta.setdefault("card", card_dict)
        for key in ("success_count", "trial_count", "delta_i"):
            if key in card_dict:
                meta.setdefault(key, card_dict[key])
    else:
        meta.setdefault("delta_i", meta.get("delta_i", 1.0))
    meta.setdefault("last_accessed", now.isoformat())
    meta.setdefault("access_count", 0)
    meta.setdefault("ema_access", 0.0)
    # Newly added memories start as shadow entries until proven useful.
    meta.setdefault("shadow", True)
    if isinstance(meta.get("summary"), str) and "summary_en" not in meta:
        meta["summary_en"] = meta["summary"]
    dyn = _get_dynamics()
    if emotional_intensity is None:
        emotional_intensity = dyn.initial_intensity
    valence = max(-1.0, min(1.0, valence))
    emotional_intensity = max(0.0, min(1.0, emotional_intensity))
    importance = max(0.0, importance)
    if last_used is None:
        last_used = now
    if success_score is None:
        success_score = 0.0
    if decay is None:
        decay = 0.0
    memory = Memory(
        memory_id=str(uuid.uuid4()),
        text=text,
        valence=valence,
        emotional_intensity=emotional_intensity,
        importance=importance,
        episode_id=episode_id,
        modality=modality,
        connections=dict(copy.deepcopy(connections)) if connections else None,
        metadata=meta,
        schema_type=schema_type,
        card=card_obj,
        created_at=now,
        memory_type=memory_type,
        pinned=pinned,
        ttl_seconds=ttl_seconds,
        last_used=last_used,
        success_score=success_score,
        decay=decay,
        verifier=verifier_obj,
    )
    if memory_type == "sensory":
        sensory_buffer.add_event(memory, ttl_seconds=ttl_seconds)
        logger.debug("Buffered sensory event %s", memory.memory_id)
        return memory

    await _wait_with_timeout(st.add_memory(memory), "st.add_memory")
    weights = adjust_weights_for_context(context)
    score = _score_decay(memory, weights)
    await _wait_with_timeout(
        st.upsert_scores([(memory.memory_id, score)]),
        "st.upsert_scores",
    )
    logger.debug("Memory %s added (%d chars).", memory.memory_id, len(text))
    return memory


async def add_lesson(
    lesson: str,
    *,
    valence: float = 0.0,
    importance: float = 0.0,
    success_score: float = 0.0,
    metadata: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Convenience wrapper to persist a lesson summary.

    The lesson is stored with ``memory_type="lesson"`` and optional emotional
    metadata describing its valence, importance and success score.
    """
    return await add(
        lesson,
        valence=valence,
        importance=importance,
        success_score=success_score,
        metadata=metadata,
        memory_type="lesson",
        store=store,
    )


async def search(
    query: str,
    *,
    retriever: str = "sparse",
    reranker: str | None = None,
    k: int = 5,
    k_bm25: int | None = None,
    k_vec: int | None = None,
    alpha: float = 0.5,
    mmr_lambda: float | None = 0.7,
    bandit: Literal["ucb", "thompson"] | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    context: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
    modality: str = "text",
    level: int | None = None,
    attention_query: str | None = None,
    summary_first: bool = False,
) -> SearchResults:
    """
    Search stored memories using configurable retrievers and rerankers.

    When ``retriever="hybrid"`` sparse and dense results are combined using
    Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    query:
        Search phrase.
    retriever:
        Retrieval strategy – ``"sparse"``, ``"vector"`` or ``"hybrid"``.
    reranker:
        Optional reranking strategy – ``"keyword"`` or ``"cross"``.
    k:
        Maximum number of results to return.
    k_bm25, k_vec:
        Candidate counts for sparse and vector searches when ``retriever="hybrid"``.
    alpha:
        Deprecated; kept for backward compatibility. No effect with RRF fusion.
    mmr_lambda:
        Balance factor for Maximal Marginal Relevance. ``None`` disables
        deduplication.
    bandit:
        Optional bandit strategy (``"ucb"`` or ``"thompson"``) used to
        incorporate historical success statistics into ranking.
    metadata_filter, context, store, modality, level:
        As in :func:`search_memory`.
    attention_query:
        Optional phrase used by rerankers; defaults to ``query``.
    summary_first:
        When ``True`` perform a two-stage lookup: first search summarised
        memories (``level > 0``) and identify the best cluster, then retrieve
        raw memories from that cluster and apply rerankers / bandits.

    Returns
    -------
    Sequence[Memory]
        Matching memories ordered by relevance.

    """
    import inspect

    def _mem_id(m: Any) -> str:
        return str(m.memory_id if hasattr(m, "memory_id") else m.id)

    cache = _get_cache()
    meta = dict(metadata_filter or {})
    meta.setdefault("modality", modality)
    ctx_tuple = tuple(sorted(context.items())) if context else None
    key = repr(
        (
            query,
            retriever,
            reranker,
            k,
            k_bm25,
            k_vec,
            alpha,
            mmr_lambda,
            bandit,
            tuple(sorted(meta.items())),
            level,
            attention_query,
            ctx_tuple,
            summary_first,
        )
    )
    cached = cache.get(key)
    if cached is not None:
        return cast("SearchResults", cached)

    st = await _resolve_store(store)

    embed_text: Callable[[str], Any] | None = None
    query_vec: Any | None = None
    if retriever in {"vector", "hybrid"} or mmr_lambda is not None:
        try:
            from embedder import embed as _embed

            embed_text = _embed
            query_vec = embed_text(query)

        except ModuleNotFoundError:
            if retriever in {"vector", "hybrid"}:
                raise
        else:
            embed_text = _embed
            query_vec = embed_text(query)

    results_list: list[Memory]
    if summary_first:
        summary_res = await search(
            query,
            retriever=retriever,
            reranker=None,
            k=1,
            k_bm25=k_bm25,
            k_vec=k_vec,
            alpha=alpha,
            mmr_lambda=mmr_lambda,
            metadata_filter=metadata_filter,
            context=context,
            store=st,
            modality=modality,
            level=1,
            attention_query=attention_query,
            summary_first=False,
        )
        if not summary_res:
            return SearchResults([], [])
        src_ids = cast("Sequence[str]", (summary_res[0].metadata or {}).get("source_ids") or [])
        if not src_ids:
            return SearchResults([], [])
        fetched = await asyncio.gather(*(st.get(mid) for mid in src_ids))
        results_list = [m for m in fetched if m is not None]
    elif retriever == "sparse":
        results = await _wait_with_timeout(
            st.search_memory(query=query, k=k, metadata_filter=meta, level=level, context=context),
            "st.search_memory",
        )
        results_list = list(results)
        for i, m in enumerate(results_list):
            meta = dict(m.metadata or {})
            meta["bm25_score"] = float(k - i)
            meta.setdefault("distance", 1.0)
            try:
                m.metadata = meta
            except Exception:  # pragma: no cover - frozen instances
                pass

    elif retriever == "vector":
        if not hasattr(st, "semantic_search"):
            raise AttributeError("Store does not support semantic_search")
        if query_vec is None:
            raise RuntimeError("Embeddings are required for vector search")
        kwargs = dict(
            embedding=query_vec.tolist(),
            k=k_vec or k,
            metadata_filter=meta,
            level=level,
            context=context,
            modality=modality,
            return_distance=True,
        )
        if mmr_lambda is not None:
            kwargs["mmr_lambda"] = mmr_lambda
        try:
            results = await _wait_with_timeout(
                st.semantic_search(**kwargs),
                "st.semantic_search",
            )
        except TypeError:  # pragma: no cover - older stores
            kwargs.pop("mmr_lambda", None)
            results = await _wait_with_timeout(
                st.semantic_search(**kwargs),
                "st.semantic_search",
            )

        results_list = []
        for item in results:
            if isinstance(item, tuple):
                m, dist = item
            else:  # pragma: no cover - older stores
                m, dist = item, 1.0
            meta = dict(m.metadata or {})
            meta["distance"] = float(dist)
            meta["bm25_score"] = 0.0
            try:
                m.metadata = meta
            except Exception:  # pragma: no cover - frozen instances
                pass
            results_list.append(m)

    elif retriever == "hybrid":
        if not hasattr(st, "semantic_search"):
            raise AttributeError("Store does not support semantic_search")
        if query_vec is None:
            raise RuntimeError("Embeddings are required for hybrid search")
        sparse_k = k_bm25 or k
        vec_k = k_vec or k

        async def _sparse() -> SearchResults:
            return await _wait_with_timeout(
                st.search_memory(
                    query=query, k=sparse_k, metadata_filter=meta, level=level, context=context
                ),
                "st.search_memory",
            )

        async def _vector() -> list[tuple[Memory, float]]:
            kwargs = dict(
                embedding=query_vec.tolist(),
                k=vec_k,
                metadata_filter=meta,
                level=level,
                context=context,
                return_distance=True,
                modality=modality,
            )

            if mmr_lambda is not None:
                kwargs["mmr_lambda"] = mmr_lambda
            try:
                res = await _wait_with_timeout(
                    st.semantic_search(**kwargs),
                    "st.semantic_search",
                )
            except TypeError:  # pragma: no cover - older stores
                kwargs.pop("mmr_lambda", None)
                res = await _wait_with_timeout(
                    st.semantic_search(**kwargs),
                    "st.semantic_search",
                )
            return cast("list[tuple[Memory, float]]", res)

        bm25_res, vec_res = await asyncio.gather(_sparse(), _vector())

        rrf_scores: defaultdict[str, float] = defaultdict(float)
        mem_map: dict[str, Memory] = {}

        for idx, m in enumerate(bm25_res):
            mid = _mem_id(m)
            rrf_scores[mid] += 1.0 / (RRF_K + idx + 1)
            mem_map[mid] = m

        for idx, (m, dist) in enumerate(vec_res):
            mid = _mem_id(m)
            rrf_scores[mid] += 1.0 / (RRF_K + idx + 1)
            meta = dict(m.metadata or {})
            meta["distance"] = float(dist)
            meta.setdefault("bm25_score", 0.0)
            try:
                m.metadata = meta
            except Exception:  # pragma: no cover - frozen instances
                pass
            mem_map[mid] = m

        for idx, m in enumerate(bm25_res):
            meta = dict(m.metadata or {})
            meta.setdefault("distance", 1.0)
            meta["bm25_score"] = float(sparse_k - idx)
            try:
                m.metadata = meta
            except Exception:  # pragma: no cover - frozen instances
                pass

        sorted_ids = sorted(mem_map, key=lambda i: rrf_scores[i], reverse=True)
        results_list = [mem_map[i] for i in sorted_ids]

    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown retriever: {retriever}")

    if mmr_lambda is not None and embed_text is not None and len(results_list) > 1:
        qvec = query_vec if query_vec is not None else embed_text(query)

        def _cos(a: Sequence[float], b: Sequence[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        items = []
        for m in results_list:
            vec = m._embedding
            if vec is None:
                vec = embed_text(m.text)
                m._embedding = vec
            items.append((m, vec))

        selected: list[tuple[Memory, Sequence[float]]] = []
        ordered: list[Memory] = []
        mmr_div: dict[str, float] = {}
        remaining = list(items)
        sims = [_cos(vec, qvec) for _, vec in remaining]
        while remaining:
            scores: list[float] = []
            divers: list[float] = []
            for i, (_m, emb) in enumerate(remaining):
                sim_q = sims[i]
                sim_s = max(_cos(emb, s_emb) for _sm, s_emb in selected) if selected else 0.0
                scores.append(mmr_lambda * sim_q - (1 - mmr_lambda) * sim_s)
                divers.append(1 - sim_s)
            idx = max(range(len(remaining)), key=lambda i: scores[i])
            mem, emb = remaining.pop(idx)
            sims.pop(idx)
            diversity = divers.pop(idx)
            selected.append((mem, emb))
            ordered.append(mem)
            mmr_div[_mem_id(mem)] = diversity
        results_list = ordered
        for mem in results_list:
            meta = dict(mem.metadata or {})
            meta["mmr_diversity"] = mmr_div.get(_mem_id(mem), 0.0)
            try:
                mem.metadata = meta
            except Exception:  # pragma: no cover - frozen instances
                pass

    # Apply optional reranker before final truncation
    rerank_query = attention_query or query
    if reranker == "keyword" and rerank_query:
        from memory_system.core.keyword_attention import order_by_attention

        weight_key: str | None = None
        for key in ("weight", "recency", "importance"):
            if any((m.metadata or {}).get(key) is not None for m in results_list):
                weight_key = key
                break
        results_list = order_by_attention(results_list, rerank_query, weight_key=weight_key)
    elif reranker == "cross" and rerank_query:
        use_cross = False
        try:  # lazy settings import
            from memory_system.settings import get_settings

            use_cross = bool(getattr(get_settings().ranking, "use_cross_encoder", False))
        except Exception:  # pragma: no cover - settings optional
            use_cross = False
        if use_cross:
            topn = min(len(results_list), 20)
            head = results_list[:topn]
            try:
                from memory_system.core.cross_reranker import order_by_cross_encoder
                from memory_system.core.store import Memory as _StoreMem

                ce_res = order_by_cross_encoder(cast("Sequence[_StoreMem]", head), rerank_query)
                if inspect.isawaitable(ce_res):
                    head = await cast("Any", ce_res)
                else:
                    head = cast("Any", ce_res)
            except Exception:  # pragma: no cover - optional dep
                head = head  # no-op on failure
            results_list = head + results_list[topn:]

    if bandit:
        from memory_system.core.bandit import rerank_with_bandit

        results_list = rerank_with_bandit(results_list, method=bandit)
    # Compute candidate scores before truncation
    try:  # lazy settings import
        from memory_system.settings import get_settings

        cfg = get_settings()
        rank = getattr(cfg, "ranking", SimpleNamespace())
        w_d = float(getattr(rank, "alpha", 1.0))
        w_s = float(getattr(rank, "beta", 0.0))
        w_f = float(getattr(rank, "w_f", 0.0))
        search_cfg = getattr(cfg, "search", SimpleNamespace())
        w_lang = float(getattr(search_cfg, "w_lang", 0.0))
    except Exception:  # pragma: no cover - settings optional
        w_d, w_s, w_f, w_lang = 1.0, 0.0, 0.0, 0.0

    try:
        from embedder import _detect_language as _detect_lang

        detect_lang = _detect_lang
    except ModuleNotFoundError:  # pragma: no cover - optional dependency

        def detect_lang(text: str) -> str:
            return "en"

    lang_q = detect_lang(query)
    for m in results_list:
        meta = dict(m.metadata or {})
        dist = float(meta.get("distance", 1.0))
        cos_sim = 1.0 - dist
        bm25 = float(meta.get("bm25_score", 0.0))
        lang_mem = meta.get("lang") or detect_lang(m.text)
        meta.setdefault("lang", lang_mem)
        schema = meta.get("schema_type") or getattr(m, "schema_type", None)
        if schema == "experience":
            success = meta.get("success_count")
            trials = meta.get("trial_count")
            card_meta = meta.get("card")
            if isinstance(card_meta, Mapping):
                success = card_meta.get("success_count", success)
                trials = card_meta.get("trial_count", trials)
            try:
                success_i = int(success) if success is not None else 0
            except (TypeError, ValueError):
                success_i = 0
            try:
                trial_i = int(trials) if trials is not None else 0
            except (TypeError, ValueError):
                trial_i = 0
            meta["p_win"] = (success_i + 1) / (trial_i + 2)
        p_win = float(meta.get("p_win", 0.0))
        score = w_d * cos_sim + w_s * bm25 + w_f * p_win
        if lang_q == lang_mem:
            score += w_lang
        meta["candidate_score"] = score
        try:
            m.metadata = meta
        except Exception:  # pragma: no cover - frozen instances
            pass

    results_list.sort(
        key=lambda m: (m.metadata or {}).get("candidate_score", 0.0),
        reverse=True,
    )
    results_list = results_list[:k]
    ctx_tags = set(context.get("tags", [])) if context else set()
    ctx_role = context.get("role") if context else None
    if ctx_tags or ctx_role is not None:
        for m in results_list:
            meta = dict(m.metadata or {})
            mem_tags = set(meta.get("tags") or [])
            tag_match = 1.0 if ctx_tags and mem_tags & ctx_tags else 0.0
            role_match = 1.0 if ctx_role is not None and meta.get("role") == ctx_role else 0.0
            meta.update({"tag_match": tag_match, "role_match": role_match})
            m.metadata = meta
    shadow_hits = [m for m in results_list if (m.metadata or {}).get("shadow")]
    primary_hits = [m for m in results_list if (m.metadata or {}).get("shadow") is not True]
    weights = adjust_weights_for_context(context)
    await _record_accesses(results_list, st, weights)
    res: SearchResults = SearchResults(primary_hits, shadow_hits)
    cache.put(key, res)
    logger.debug("Search for '%s' returned %d result(s).", query, len(results_list))
    return res


async def delete(
    memory_id: str,
    *,
    store: MemoryStoreProtocol | None = None,
) -> None:
    """
    Delete a memory by ``memory_id`` if it exists.

    Args:
        memory_id (str): The memory identifier.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    """
    st = await _resolve_store(store)
    await _wait_with_timeout(st.delete_memory(memory_id), "st.delete_memory")
    _get_cache().clear()
    logger.debug("Memory %s deleted.", memory_id)


async def update(
    memory_id: str,
    *,
    text: str | None = None,
    metadata: MutableMapping[str, Any] | None = None,
    importance: float | None = None,
    importance_delta: float | None = None,
    valence: float | None = None,
    valence_delta: float | None = None,
    emotional_intensity: float | None = None,
    emotional_intensity_delta: float | None = None,
    memory_type: (
        Literal["sensory", "working", "episodic", "semantic", "skill", "lesson"] | None
    ) = None,
    pinned: bool | None = None,
    ttl_seconds: int | None = None,
    last_used: _dt.datetime | None = None,
    success_score: float | None = None,
    decay: float | None = None,
    context: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Update text, metadata or scoring fields of an existing memory.

    This helper allows changing the ``importance`` score directly without
    calling :func:`reinforce` by supplying either ``importance`` or
    ``importance_delta``.

    ``last_accessed`` in the memory's metadata is automatically set to the
    current UTC timestamp.  Absolute ``valence`` and ``emotional_intensity``
    values may be supplied to overwrite existing ones while the ``*_delta``
    counterparts adjust the stored values.

    Args:
        memory_id (str): The memory identifier.
        text (str | None, optional): New text. Defaults to None.
        metadata (MutableMapping[str, Any] | None, optional): New metadata. Defaults to None.
        importance (float | None, optional): New importance value. Defaults to None.
        importance_delta (float | None, optional): Increment for importance. Defaults to None.
        valence (float | None, optional): New emotional valence. Defaults to None.
        valence_delta (float | None, optional): Increment for emotional valence.
            Defaults to None.
        emotional_intensity (float | None, optional): New emotional intensity.
            Defaults to None.
        emotional_intensity_delta (float | None, optional): Increment for
            emotional intensity. Defaults to None.
        context (MutableMapping[str, Any] | None, optional): Context influencing ranking weights.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The updated memory object.

    """
    st = await _resolve_store(store)
    meta: MutableMapping[str, Any] = dict(copy.deepcopy(metadata) if metadata else {})
    meta["last_accessed"] = _dt.datetime.now(_dt.UTC).isoformat()
    updated = await _wait_with_timeout(
        st.update_memory(
            memory_id,
            text=text,
            metadata=meta,
            importance=importance,
            importance_delta=importance_delta,
            valence=valence,
            valence_delta=valence_delta,
            emotional_intensity=emotional_intensity,
            emotional_intensity_delta=emotional_intensity_delta,
            memory_type=memory_type,
            pinned=pinned,
            ttl_seconds=ttl_seconds,
            last_used=last_used,
            success_score=success_score,
            decay=decay,
        ),
        "st.update_memory",
    )
    raw_count = (updated.metadata or {}).get("access_count")
    count = int(raw_count) + 1 if raw_count is not None else 1
    updated = await _wait_with_timeout(
        st.update_memory(memory_id, metadata={"access_count": count}),
        "st.update_memory",
    )
    weights = adjust_weights_for_context(context)
    score = _score_decay(updated, weights)
    await _wait_with_timeout(
        st.upsert_scores([(memory_id, score)]),
        "st.upsert_scores",
    )
    _get_cache().clear()
    logger.debug("Memory %s updated.", memory_id)
    return updated


async def _verify_memory(mem: Memory) -> bool:
    """Run type specific verifier if present."""
    verifier = (mem.metadata or {}).get("verifier")
    if not verifier:
        return True
    vtype = verifier.get("type")
    if vtype == "code":
        path = verifier.get("test_suite_path")
        if not path:
            return False
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest",
                path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False
    if vtype == "math":
        check = verifier.get("check")
        if not check:
            return False
        try:
            func = eval(check)
        except Exception:
            return False
        try:
            return bool(func(mem.text))
        except Exception:
            return False
    return False


async def update_trust_scores(
    task: str,
    top_memories: Sequence[Memory],
    reasoner: Callable[[str, Sequence[Memory]], Awaitable[float]],
    *,
    store: MemoryStoreProtocol | None = None,
) -> None:
    """
    Approximate each memory's contribution and persist ``trust_score``.

    The ``reasoner`` callable should accept ``(task, memories)`` and return a
    numeric success metric where higher values indicate better performance. For
    each memory in ``top_memories`` the reasoning is rerun with that memory
    removed and the drop in the metric is added to its ``trust_score``
    metadata.
    """
    if not top_memories:
        return
    baseline = await reasoner(task, top_memories)
    for mem in top_memories:
        subset = [m for m in top_memories if m.memory_id != mem.memory_id]
        score = await reasoner(task, subset)
        contribution = baseline - score
        prev = float((mem.metadata or {}).get("trust_score", 0.0))
        new_score = prev + contribution
        meta: dict[str, Any] = {"trust_score": new_score}
        # Promote shadow memories once they have proven useful enough and pass verification.
        if new_score >= SHADOW_PROMOTION_THRESHOLD and (mem.metadata or {}).get("shadow", True):
            if await _verify_memory(mem):
                meta["shadow"] = False
        await update(
            mem.memory_id,
            metadata=meta,
            store=store,
        )


async def record_search_feedback(
    memory: Memory,
    success: bool,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Persist success/failure statistics for ``memory`` and enqueue it for retraining.

    Each call updates trial and success counters and recomputes the quality
    error ``delta_i = 1 - success_rate``. Updated memories are appended to a
    queue which is sampled with weights ``(abs(delta_i) + ε) ** α``. The
    updated :class:`Memory` instance is returned.
    """
    st = await _resolve_store(store)
    meta = dict(memory.metadata or {})
    trials_raw = meta.get("trial_count")
    trials = int(trials_raw) + 1 if trials_raw is not None else 1
    successes_raw = meta.get("success_count")
    successes = (int(successes_raw) if successes_raw is not None else 0) + (1 if success else 0)
    delta = 1.0 - (successes / trials if trials else 0.0)
    meta.update({"trial_count": trials, "success_count": successes, "delta_i": delta})
    card_meta = meta.get("card")
    if isinstance(card_meta, Mapping):
        card_meta = dict(card_meta)
        card_meta.update({"trial_count": trials, "success_count": successes, "delta_i": delta})
        meta["card"] = card_meta
    updated = await _wait_with_timeout(
        st.update_memory(memory.memory_id, metadata=meta),
        "st.update_memory",
    )
    _feedback_heap.append(updated)
    memory.metadata = updated.metadata
    return updated


async def reinforce(
    memory_id: str,
    amount: float | None = None,
    *,
    valence_delta: float | None = None,
    intensity_delta: float | None = None,
    context: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Reinforce a memory's importance and optionally its emotional context.

    The memory's metadata receives an updated ``last_accessed`` timestamp.

    By default only the ``importance`` field is reinforced. Supplying
    ``valence_delta`` and/or ``intensity_delta`` applies the same deltas to
    the respective ``valence`` and ``emotional_intensity`` attributes.

    Args:
        memory_id (str): The memory identifier.
        amount (float | None, optional): Importance increment. Defaults to configured value.
        valence_delta (float | None, optional): Change applied to ``valence``.
        intensity_delta (float | None, optional): Change applied to
            ``emotional_intensity``.
        context (MutableMapping[str, Any] | None, optional): Context influencing ranking weights.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        Memory: The reinforced memory object.

    """
    st = await _resolve_store(store)
    meta = {"last_accessed": _dt.datetime.now(_dt.UTC).isoformat()}
    dyn = _get_dynamics()
    if amount is None:
        amount = dyn.reinforce_delta
    amount = max(-1.0, min(1.0, amount))
    if valence_delta is not None:
        valence_delta = max(-1.0, min(1.0, valence_delta))
    if intensity_delta is not None:
        intensity_delta = max(-1.0, min(1.0, intensity_delta))
    updated = await _wait_with_timeout(
        st.update_memory(
            memory_id,
            importance_delta=amount,
            valence_delta=valence_delta,
            emotional_intensity_delta=intensity_delta,
            metadata=meta,
        ),
        "st.update_memory",
    )
    raw_count = (updated.metadata or {}).get("access_count")
    count = int(raw_count) + 1 if raw_count is not None else 1
    updated = await _wait_with_timeout(
        st.update_memory(memory_id, metadata={"access_count": count}),
        "st.update_memory",
    )
    weights = adjust_weights_for_context(context)
    score = _score_decay(updated, weights)
    await _wait_with_timeout(
        st.upsert_scores([(memory_id, score)]),
        "st.upsert_scores",
    )
    logger.debug("Memory %s reinforced by %.2f.", memory_id, amount)
    return updated


async def list_recent(
    n: int = 20,
    *,
    level: int | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    store: MemoryStoreProtocol | None = None,
) -> list[Memory]:
    """
    Return *n* most recently added memories in descending chronological order.

    Args:
        n (int, optional): Number of memories. Defaults to 20.
        level (int | None, optional): Exact level filter. Defaults to None.
        metadata_filter (MutableMapping[str, Any] | None, optional): Additional metadata filters.
        store (MemoryStoreProtocol | None, optional): Store object. Defaults to None.

    Returns:
        list[Memory]: List of recent memories.

    """
    st = await _resolve_store(store)
    recent_seq = await _wait_with_timeout(
        st.list_recent(n=n, level=level, metadata_filter=metadata_filter),
        "st.list_recent",
    )
    recent = list(recent_seq)
    logger.debug("Fetched %d recent memories.", len(recent))
    return recent


# Weights used when ranking memories via :func:`list_best`.
# Positive valence is treated as a benefit while negative valence is
# penalised with a reduced weight so that strongly negative memories
# need additional importance or intensity to surface.


@dataclass(slots=True)
class ListBestWeights:
    """
    Configuration for weighting memory attributes when ranking.

    The overall score is computed as::

        score = w_valence * valence + w_importance * importance
                + w_recency * exp(-λΔt) + w_freq * log(1 + ema_access)
                + w_tag * tag_match + w_role * role_match

    where Δt is the time since the memory was last accessed and
    ``ema_access`` is an exponentially decayed access counter.
    """

    importance: float = 1.0
    emotional_intensity: float = 1.0
    valence_pos: float = 1.0
    valence_neg: float = 0.5
    recency: float = 1.0
    frequency: float = 1.0
    tag_match: float = 1.0
    role_match: float = 1.0


def _get_ranking_weights() -> ListBestWeights:
    """Load ranking weights from configuration if available."""
    try:  # lazy import to avoid optional dependency at import time
        from memory_system.settings import get_settings

        cfg = get_settings()
        data = cfg.ranking.model_dump()
        data.pop("min_score", None)
        data.pop("adaptation", None)
        data.pop("use_cross_encoder", None)
        data.pop("alpha", None)
        data.pop("beta", None)
        data.pop("gamma", None)
        data.pop("w_f", None)
        return ListBestWeights(**data)
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - settings module optional
        return ListBestWeights()


def adjust_weights_for_context(
    context: MutableMapping[str, Any] | None = None,
) -> ListBestWeights:
    """
    Return ranking weights adjusted for the provided ``context``.

    The base weights are loaded via :func:`_get_ranking_weights` and then
    modified according to any adaptation rules defined in configuration.
    ``context`` is expected to be a mapping of simple key/value pairs.
    """
    weights = _get_ranking_weights()
    if not context:
        return weights
    try:  # pragma: no cover - settings module optional
        from memory_system.settings import get_settings

        rules = getattr(get_settings().ranking, "adaptation", {})
    except (ImportError, AttributeError):
        return weights
    for key, value in context.items():
        try:
            key_rules = rules.get(key, {})
            val_rule = key_rules.get(str(value))
            if not isinstance(val_rule, dict):
                continue
            for field, val in val_rule.items():
                if hasattr(weights, field):
                    try:
                        setattr(weights, field, float(val))
                    except (TypeError, ValueError):
                        continue
        except AttributeError:
            continue
    return weights


def _score_best(m: Memory, weights: ListBestWeights) -> float:
    """
    Return the ranking score for a memory.

    Implements the formula documented in :class:`ListBestWeights`::

        score = w_valence * valence + w_importance * importance
                + w_recency * exp(-λΔt) + w_freq * log(1 + ema_access)
                + w_tag * tag_match + w_role * role_match

    ``Δt`` is measured in seconds since the memory was last accessed and
    ``ema_access`` defaults to ``0`` when unspecified.
    """
    valence_weight = weights.valence_pos if m.valence >= 0 else weights.valence_neg
    ema = 0.0
    if m.metadata:
        try:
            ema = float(m.metadata.get("ema_access", 0.0))
        except (ValueError, TypeError):
            ema = 0.0
    last = last_accessed(m)
    now = _dt.datetime.now(_dt.UTC)
    delta_seconds = max(0.0, (now - last).total_seconds())
    # Decay constant chosen so that one day results in ~e^{-1} multiplier.
    lambda_ = 1.0 / 86_400.0
    recency_term = weights.recency * math.exp(-lambda_ * delta_seconds)
    freq_term = weights.frequency * math.log1p(ema)
    tag = 0.0
    role = 0.0
    if m.metadata:
        try:
            tag = float(m.metadata.get("tag_match", 0.0))
        except (ValueError, TypeError):
            tag = 0.0
        try:
            role = float(m.metadata.get("role_match", 0.0))
        except (ValueError, TypeError):
            role = 0.0
    return (
        valence_weight * m.valence
        + weights.importance * m.importance
        + recency_term
        + freq_term
        + weights.tag_match * tag
        + weights.role_match * role
    )


class _MemoryLike(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any] | None: ...

    @property
    def created_at(self) -> _dt.datetime: ...


def last_accessed(m: _MemoryLike) -> _dt.datetime:
    """
    Return the last accessed timestamp for *m*.

    Falls back to ``m.created_at`` when the ``metadata`` lacks the
    ``last_accessed`` key or contains an invalid timestamp.
    """
    if m.metadata:
        ts = m.metadata.get("last_accessed")
        if isinstance(ts, str):
            try:
                return _dt.datetime.fromisoformat(ts)
            except ValueError:
                pass
    return m.created_at


def _score_decay(m: Memory, weights: ListBestWeights) -> float:
    """Return a ranking score for *m* with recency and EMA."""
    last = last_accessed(m)
    now = _dt.datetime.now(_dt.UTC)
    delta = max(0.0, (now - last).total_seconds())
    age_days = delta / 86_400.0
    dyn = _get_dynamics()
    rate = max(cast("float", getattr(dyn, "decay_rate", 0.0)), EPSILON)
    decay_law = cast("str", getattr(dyn, "decay_law", "exponential"))
    if decay_law == "logarithmic":
        decay = 1.0 / (1.0 + math.log1p(age_days) / rate)
    else:
        decay = math.exp(-age_days / rate)
    decayed_intensity = m.emotional_intensity * float(decay)
    ema = 0.0
    if m.metadata:
        try:
            ema = float(m.metadata.get("ema_access", 0.0))
        except (ValueError, TypeError):
            ema = 0.0
    tag = 0.0
    role = 0.0
    if m.metadata:
        try:
            tag = float(m.metadata.get("tag_match", 0.0))
        except (ValueError, TypeError):
            tag = 0.0
        try:
            role = float(m.metadata.get("role_match", 0.0))
        except (ValueError, TypeError):
            role = 0.0
    valence_weight = weights.valence_pos if m.valence >= 0 else weights.valence_neg
    recency = math.exp(-delta / RECENCY_TAU)
    base = (
        weights.importance * m.importance
        + weights.emotional_intensity * decayed_intensity
        + valence_weight * m.valence
        + weights.tag_match * tag
        + weights.role_match * role
    )
    freq_term = weights.frequency * math.log1p(ema)
    return base * recency + freq_term


async def _record_accesses(
    memories: Sequence[Memory],
    st: MemoryStoreProtocol,
    weights: ListBestWeights,
) -> None:
    """Update access metadata and scores for *memories*."""
    if not memories or not hasattr(st, "update_memory") or not hasattr(st, "upsert_scores"):
        return
    now = _dt.datetime.now(_dt.UTC)
    updates = []
    scores: list[tuple[str, float]] = []
    for m in memories:
        last = last_accessed(m)
        delta = max(0.0, (now - last).total_seconds())
        prev_meta = dict(m.metadata or {})
        try:
            ema_prev = float(prev_meta.get("ema_access", 0.0))
        except (ValueError, TypeError):
            ema_prev = 0.0
        ema = ema_prev * math.exp(-delta / RECENCY_TAU) + 1.0
        raw_prev = prev_meta.get("access_count")
        if raw_prev is not None:
            try:
                count_prev = int(raw_prev)
            except (ValueError, TypeError):
                count_prev = 0
        else:
            count_prev = 0
        meta = {
            "last_accessed": now.isoformat(),
            "access_count": count_prev + 1,
            "ema_access": ema,
        }
        ttl_update: int | None = None
        if m.ttl_seconds is not None and secrets.randbelow(1000) / 1000 < TTL_BUMP_PROB:
            ttl_update = int(m.ttl_seconds * TTL_BUMP_FACTOR)
            m.ttl_seconds = ttl_update
        kwargs: dict[str, Any] = {"metadata": meta}
        if ttl_update is not None:
            kwargs["ttl_seconds"] = ttl_update
        updates.append(
            _wait_with_timeout(
                st.update_memory(m.memory_id, **kwargs),
                "st.update_memory",
            )
        )
        prev_meta.update(meta)
        # ``Memory`` instances are frozen dataclasses, so we bypass the normal
        # attribute assignment when updating metadata.  This mirrors how the
        # real store would return mutable objects but keeps our lightweight
        # model compatible with the tests.
        object.__setattr__(m, "metadata", prev_meta)
        scores.append((m.memory_id, _score_decay(m, weights)))
    await asyncio.gather(*updates)
    await _wait_with_timeout(st.upsert_scores(scores), "st.upsert_scores")


def _ensure_memory(m: Any) -> Memory:
    """Coerce *m* into a :class:`Memory` instance if needed."""
    if isinstance(m, Memory):
        return m
    meta = getattr(m, "metadata", None)
    schema_type = getattr(m, "schema_type", None)
    if schema_type is None and meta:
        schema_type = meta.get("schema_type")
    meta_dict = dict(meta) if meta is not None else None
    if schema_type == "experience" and meta_dict is not None:
        success = meta_dict.get("success_count")
        trials = meta_dict.get("trial_count")
        card_meta = meta_dict.get("card")
        if isinstance(card_meta, Mapping):
            success = card_meta.get("success_count", success)
            trials = card_meta.get("trial_count", trials)
        try:
            success_i = int(success) if success is not None else 0
        except (TypeError, ValueError):
            success_i = 0
        try:
            trial_i = int(trials) if trials is not None else 0
        except (TypeError, ValueError):
            trial_i = 0
        meta_dict["p_win"] = (success_i + 1) / (trial_i + 2)
    card_data = getattr(m, "card", None)
    if card_data is None and meta:
        card_data = meta.get("card")
    card_obj: ExperienceCard | None = None
    if isinstance(card_data, ExperienceCard):
        card_obj = card_data
    elif isinstance(card_data, Mapping):
        card_obj = ExperienceCard(**card_data)
    return Memory(
        memory_id=getattr(m, "memory_id", m.id),
        text=m.text,
        created_at=m.created_at,
        valence=getattr(m, "valence", 0.0),
        emotional_intensity=getattr(m, "emotional_intensity", 0.0),
        importance=getattr(m, "importance", 0.0),
        episode_id=getattr(m, "episode_id", None),
        modality=getattr(m, "modality", "text"),
        connections=getattr(m, "connections", None),
        metadata=meta_dict,
        schema_type=schema_type,
        card=card_obj,
        memory_type=getattr(m, "memory_type", "episodic"),
        pinned=getattr(m, "pinned", False),
        ttl_seconds=getattr(m, "ttl_seconds", None),
        last_used=getattr(m, "last_used", None),
        success_score=getattr(m, "success_score", None),
        decay=getattr(m, "decay", None),
    )


async def list_best(
    n: int = 5,
    *,
    store: MemoryStoreProtocol | None = None,
    level: int | None = None,
    metadata_filter: MutableMapping[str, Any] | None = None,
    weights: ListBestWeights | None = None,
) -> list[Memory]:
    """
    Return the `n` memories ranked by score.

    Parameters
    ----------
    n:
        Number of memories to return.
    store:
        Optional explicit memory store.
    level:
        Optional level filter applied before ranking.
    metadata_filter:
        Optional metadata constraints (exact match) applied before ranking.
    weights:
        If ``None`` (default) the store returns memories ordered by its
        precomputed ``memory_scores`` table.  When provided, the store
        computes the ranking on the fly using these weighting coefficients.

    """
    st = await _resolve_store(store)
    candidates = await _wait_with_timeout(
        st.top_n_by_score(
            n,
            level=level,
            metadata_filter=metadata_filter,
            weights=weights,
        ),
        "st.top_n_by_score",
    )

    results = [_ensure_memory(m) for m in candidates]
    weight_obj = weights or _get_ranking_weights()
    await _record_accesses(results, st, weight_obj)
    return results


async def promote_personal(
    title: str,
    claim: str,
    sources: Sequence[str],
    license: str,
    *,
    store: MemoryStoreProtocol | None = None,
) -> Memory:
    """
    Promote personal knowledge into the global store.

    The function evaluates novelty, demand, reliability and performs
    PII/licence checks before persisting the *claim* as a semantic memory.

    Parameters
    ----------
    title:
        Human friendly title for the promoted card.
    claim:
        The statement to store.
    sources:
        Iterable of evidence URLs supporting the claim.
    license:
        License under which the content is shared.
    store:
        Optional explicit store; defaults to the process wide store.

    Returns
    -------
    Memory
        The stored memory record.

    Raises
    ------
    ValueError
        If any validation check fails.

    """
    if not claim:
        raise ValueError("claim must not be empty")

    st = await _resolve_store(store)

    # PII check
    if EnhancedPIIFilter().detect(claim):
        raise ValueError("PII detected in claim")

    # License check – accept a small whitelist of licences
    allowed = {"mit", "apache-2.0", "bsd-3-clause", "cc0", "cc-by", "cc-by-sa", "public-domain"}
    if license.lower() not in allowed:
        raise ValueError("unsupported license")

    # Novelty check – ensure claim not already stored
    try:
        existing = await st.search_memory(query=claim, k=1)
    except Exception:
        existing = SearchResults([], [])
    if existing:
        raise ValueError("claim already exists")

    # Demand + reliability – require HTTP sources
    if not sources:
        raise ValueError("no sources provided")
    if any(not (s.startswith("http://") or s.startswith("https://")) for s in sources):
        raise ValueError("unreliable sources")

    # HEAD validation of sources and timestamp
    httpx = require_httpx()
    now = _dt.datetime.now(tz=_dt.UTC)
    validation: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for src in sources:
            ok = True
            try:
                resp = await client.request("HEAD", src)
                ok = resp.status_code < 400
            except Exception:
                ok = False
            validation[src] = {"ok": ok, "checked_at": now.isoformat()}

    valid_sources = [s for s, info in validation.items() if info["ok"]]
    trust_score = min(1.0, len(valid_sources) / 3)
    ttl_seconds: int | None = None
    if len(valid_sources) < len(sources):
        if len(sources) > 0:
            trust_score *= len(valid_sources) / len(sources)
        ttl_seconds = STALE_TTL_SECONDS

    # Contradiction counters
    contradiction_sources: set[str] = set()
    try:
        existing_title = await st.search_memory(query=title, k=5)
    except Exception:
        existing_title = SearchResults([], [])
    for m in existing_title:
        m_meta = m.metadata or {}
        card = m_meta.get("card") or {}
        if card.get("title") == title and card.get("claim") != claim:
            for src in card.get("sources", []):
                if src in sources:
                    SOURCE_CONTRADICTIONS[src] += 1
                    contradiction_sources.add(src)

    quarantined = any(
        SOURCE_CONTRADICTIONS[s] >= CONTRADICTION_THRESHOLD for s in contradiction_sources
    )

    card = {
        "title": title,
        "claim": claim,
        "sources": list(sources),
        "license": license,
        "trust_score": trust_score,
        "validation": validation,
    }
    metadata: dict[str, Any] = {"card": card, "trust_score": trust_score}
    if quarantined:
        metadata["quarantined"] = True

    return await add(
        claim,
        metadata=metadata,
        memory_type="semantic",
        ttl_seconds=ttl_seconds,
        store=st,
    )


__all__ = [
    "DuplicateMemoryError",
    "ExperienceCard",
    "ListBestWeights",
    "Memory",
    "MemoryStoreProtocol",
    "PersonalCard",
    "add",
    "add_lesson",
    "adjust_weights_for_context",
    "delete",
    "get_default_store",
    "last_accessed",
    "list_best",
    "list_recent",
    "promote_personal",
    "record_search_feedback",
    "reinforce",
    "sample_feedback",
    "search",
    "set_default_store",
    "update",
]
