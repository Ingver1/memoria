"""
maintenance.py — Long-term memory maintenance utilities.

This module provides helpers to:
  • cluster similar memories (by cosine similarity on text embeddings),
  • produce short “summary memories” for each cluster (consolidation),
  • forget low-value memories using an age-aware decay score.

All functions are framework-agnostic and work with the existing
SQLite store and FAISS HNSW index.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import heapq
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from embedder import embed as embed_text
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.event_log import EventLog
from memory_system.core.factify import factify
from memory_system.core.hierarchical_summarizer import HierarchicalSummarizer
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.interfaces import VectorIndexMaintenance
from memory_system.core.memory_dynamics import _age_decay
from memory_system.core.store import Memory, SQLiteMemoryStore, _normalize_for_hash
from memory_system.core.summarization import STRATEGIES, SummaryStrategy
from memory_system.settings import UnifiedSettings
from memory_system.utils.blake import blake3_hex
from memory_system.utils.dependencies import require_faiss
from memory_system.utils.metrics import (
    CONSOLIDATIONS_TOTAL,
    DRIFT_COUNT,
    HEAL_COUNT,
    LAT_CONSOLIDATION,
    LAT_FORGET,
    MEM_CREATED_TOTAL,
    MEM_DELETED_TOTAL,
    PURGED_TOTAL,
    measure_time_async,
)

log = logging.getLogger(__name__)

# ----------------------------- small utils -----------------------------


def _cos_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity for two 1D embeddings (assumed L2-normalized)."""
    return float(np.dot(a, b))


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _decay_score(
    *,
    importance: float,
    valence: float,
    emotional_intensity: float,
    age_days: float,
    trust: float = 1.0,
    error_flag: bool = False,
) -> float:
    """Return decay score factoring trust and errors."""
    importance = max(0.0, min(1.0, importance))
    valence = max(-1.0, min(1.0, valence))
    emotional_intensity = max(0.0, min(1.0, emotional_intensity))
    trust = max(0.0, min(1.0, trust))
    base = importance * valence * emotional_intensity * trust
    if error_flag:
        base *= 0.5
    base = max(0.0, base)
    return base * _age_decay(age_days)


def _update_env_file(values: dict[str, int]) -> None:
    path = Path(".env")
    data: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    data.update({k: str(v) for k, v in values.items()})
    path.write_text("\n".join(f"{k}={v}" for k, v in data.items()) + "\n")


def autotune_index(index: FaissHNSWIndex, sample_size: int = 100) -> None:
    try:
        faiss = require_faiss()
    except Exception as exc:  # pragma: no cover - make import error explicit
        raise ModuleNotFoundError("faiss is required for autotuning") from exc
    if not hasattr(index, "_id_map") or not hasattr(index, "index"):
        return
    ids = list(index._id_map.keys())
    if not ids:
        return
    k = min(len(ids), sample_size)
    sample_ids = random.sample(ids, k)
    try:
        vecs = np.vstack([index.index.reconstruct(int(i)) for i in sample_ids]).astype(np.float32)
    except Exception:
        return
    M, ef_c, ef_s = index.auto_tune(vecs)
    base = faiss.downcast_index(index.index.index)
    if hasattr(base, "hnsw"):
        base.hnsw.efConstruction = ef_c
        base.hnsw.efSearch = ef_s
    index.ef_search = ef_s
    _update_env_file({"AI_FAISS__M": M, "AI_FAISS__EF_SEARCH": ef_s, "AI_FAISS__NPROBE": ef_s})


async def check_and_heal_index_drift(
    store: SQLiteMemoryStore,
    index: VectorIndexMaintenance,
    *,
    chunk_size: int = 1_000,
    drift_threshold: int = 100,
) -> tuple[int, int]:
    """
    Synchronise IDs between SQLite and vector index.

    Parameters
    ----------
    store:
        Metadata store containing the canonical set of memories.
    index:
        Vector index (FAISS or Qdrant) storing embeddings.
    chunk_size:
        Number of rows fetched per iteration from the database.
    drift_threshold:
        Emit a warning when the total drift exceeds this value.

    Returns
    -------
    tuple[int, int]
        Number of missing and dangling IDs respectively.

    """
    mem_map: dict[str, Memory] = {}
    async for batch in store.search_iter(limit=None, chunk_size=chunk_size):
        for mem in batch:
            mem_map[mem.id] = mem

    db_ids = set(mem_map)
    idx_ids: set[str]

    id_map = getattr(index, "_id_map", None)
    if id_map:
        idx_ids = set(id_map.values())
    else:
        try:
            ids = index.list_ids()
            if asyncio.iscoroutine(ids):
                ids = await ids
            idx_ids = set(cast("Sequence[str]", ids))
        except Exception:  # pragma: no cover - defensive
            log.exception("failed to list ids from index")
            idx_ids = set()

    missing = db_ids - idx_ids
    dangling = idx_ids - db_ids
    drift = len(missing) + len(dangling)
    if drift:
        log.info("index drift detected: missing=%d dangling=%d", len(missing), len(dangling))
        if drift > drift_threshold:
            log.warning("index drift %d exceeds threshold %d", drift, drift_threshold)
    DRIFT_COUNT.inc(drift)

    healed = 0
    if missing:
        ids_to_add = list(missing)
        vectors = np.vstack([embed_text(mem_map[i].text) for i in ids_to_add]).astype(np.float32)
        try:
            index.add_vectors(ids_to_add, vectors)
            healed += len(ids_to_add)
        except Exception:  # pragma: no cover - defensive
            log.exception("failed to reindex missing memories")

    if dangling:
        ids_to_remove = list(dangling)
        try:
            try:
                index.remove_ids(ids_to_remove)
            except AttributeError:
                await index.delete(ids_to_remove)
            healed += len(ids_to_remove)
        except Exception:  # pragma: no cover - defensive
            log.exception("failed to remove dangling ids")

    if healed:
        HEAL_COUNT.inc(healed)

    return len(missing), len(dangling)


# Backwards compatibility alias
heal_index_drift = check_and_heal_index_drift


# ----------------------------- clustering -----------------------------


def cluster_memories(
    embeddings: Sequence[NDArray[np.float32]],
    threshold: float = 0.83,
) -> list[list[int]]:
    """
    Greedy single-pass clustering by cosine similarity to the first item in
    each cluster. O(n²) in the worst case but simple and robust for small/medium
    collections.

    Returns a list of clusters; each cluster is a list of indices into
    `embeddings`.
    """
    n = len(embeddings)
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        seed = embeddings[i]
        group = [i]
        for j in range(i + 1, n):
            if not visited[j] and _cos_sim(seed, embeddings[j]) >= threshold:
                visited[j] = True
                group.append(j)
        clusters.append(group)

    return clusters


def cluster_memories_faiss(
    embeddings: Sequence[NDArray[np.float32]],
    threshold: float = 0.83,
    *,
    n_clusters: int | None = None,
) -> list[list[int]]:
    """
    Cluster embeddings using FAISS k-means for better scalability.

    Parameters
    ----------
    embeddings:
        Sequence of L2-normalised embedding vectors.
    threshold:
        Minimum cosine similarity to the cluster centroid to remain in the
        cluster.  Items below this threshold form their own clusters.
    n_clusters:
        Optional explicit number of clusters.  When omitted a heuristic based on
        ``sqrt(n)`` is used.

    """
    if not embeddings:
        return []

    faiss = require_faiss()
    x = np.vstack(embeddings).astype("float32")
    n, dim = x.shape
    faiss.normalize_L2(x)

    if n_clusters is None:
        n_clusters = max(1, int(np.sqrt(n)))

    kmeans = faiss.Kmeans(dim, n_clusters, niter=20, verbose=False, spherical=True)
    kmeans.train(x)
    centroids = np.asarray(faiss.vector_to_array(kmeans.centroids), dtype=np.float32).reshape(
        n_clusters, dim
    )
    _, assign = kmeans.index.search(x, 1)

    clusters: list[list[int]] = [[] for _ in range(n_clusters)]
    for idx, cid in enumerate(assign.ravel()):
        if _cos_sim(x[idx], centroids[cid]) >= threshold:
            clusters[cid].append(idx)
        else:
            clusters.append([idx])

    return [c for c in clusters if c]


def cluster_memories_auto(
    embeddings: Sequence[NDArray[np.float32]],
    threshold: float = 0.83,
    *,
    algorithm: str = "auto",
    large_threshold: int = 1_000,
) -> list[list[int]]:
    """
    Dispatch to an appropriate clustering algorithm.

    Parameters
    ----------
    algorithm:
        ``"greedy"`` forces the O(n²) algorithm, ``"faiss"`` forces the scalable
        FAISS-based method and ``"auto"`` selects based on ``large_threshold``.
    large_threshold:
        When ``algorithm`` is ``"auto"`` the FAISS method is used if the number
        of embeddings exceeds this value.

    """
    algo = algorithm.lower()
    if algo == "greedy":
        return cluster_memories(embeddings, threshold)
    if algo == "faiss":
        return cluster_memories_faiss(embeddings, threshold)
    if algo == "auto":
        if len(embeddings) > large_threshold:
            try:
                return cluster_memories_faiss(embeddings, threshold)
            except ModuleNotFoundError:
                pass
        return cluster_memories(embeddings, threshold)
    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def summarize_cluster(
    memories: Sequence[Memory],
    *,
    strategy: str | SummaryStrategy = "head2tail",
) -> tuple[str, float, float, float, float]:
    """
    Summarise a cluster of memories and compute aggregated attributes.

    Parameters
    ----------
    memories:
        Sequence of :class:`Memory` objects in the cluster.
    strategy:
        Strategy name or callable controlling how texts are summarised.
        The default ``"head2tail"`` joins the two most important texts with an
        ellipsis.  Custom callables may be supplied for experimentation.

    """
    if not memories:
        return "", 0.0, 0.0, 0.0, 0.0

    if isinstance(strategy, str):
        try:
            fn = STRATEGIES[strategy]
        except KeyError as exc:
            raise ValueError(f"Unknown summary strategy: {strategy}") from exc
    else:
        fn = strategy

    summary_text = fn(memories)
    avg_importance = float(np.mean([m.importance for m in memories]))
    avg_valence = float(np.mean([m.valence for m in memories]))
    avg_intensity = float(np.mean([m.emotional_intensity for m in memories]))
    avg_trust = float(np.mean([(m.metadata or {}).get("trust_score", 0.0) for m in memories]))
    return summary_text, avg_importance, avg_valence, avg_intensity, avg_trust


# ------------------------ consolidation / forgetting -------------------


@measure_time_async(LAT_CONSOLIDATION)
async def consolidate_store(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    threshold: float = 0.83,
    max_fetch: int = 100_000,
    strategy: str | SummaryStrategy = "head2tail",
    chunk_size: int = 1_000,
    min_level: int | None = None,
    final: bool | None = None,
    cluster_algorithm: str | None = None,
    cluster_auto_threshold: int | None = None,
) -> list[Memory]:
    """
    Cluster similar memories, create a summary memory per cluster, then delete
    the originals that were consolidated.

    Parameters
    ----------
    min_level:
        If provided, only process memories with ``level >= min_level``.
    final:
        If ``True`` or ``False``, restrict to memories where
        ``metadata['final']`` matches the value. ``False`` also includes
        memories without the ``final`` flag.

    Returns
    -------
    list[Memory]
        The summary memories that were created.

    """
    created: list[Memory] = []
    ids_to_remove: list[str] = []
    modality_map: dict[str, str] = {}
    pending_adds: list[tuple[Memory, NDArray[np.float32]]] = []
    event_log = EventLog(store._dsn)

    if cluster_algorithm is None or cluster_auto_threshold is None:
        try:  # pragma: no cover - settings lookup is straightforward
            from memory_system.settings import get_settings

            cfg = get_settings()
            cluster_algorithm = cluster_algorithm or getattr(
                cfg.maintenance, "cluster_algorithm", "auto"
            )
            cluster_auto_threshold = cluster_auto_threshold or getattr(
                cfg.maintenance, "cluster_auto_threshold", 1_000
            )
        except Exception:
            cluster_algorithm = cluster_algorithm or "auto"
            cluster_auto_threshold = cluster_auto_threshold or 1_000

    async for chunk in store.search_iter(
        limit=max_fetch,
        chunk_size=chunk_size,
        min_level=min_level,
        final=final,
    ):
        texts = [m.text for m in chunk]
        embeddings = embed_text(texts)
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        clusters = cluster_memories_auto(
            [embeddings[i] for i in range(embeddings.shape[0])],
            threshold,
            algorithm=cluster_algorithm or "auto",
            large_threshold=cluster_auto_threshold or 1_000,
        )

        for group in clusters:
            if len(group) <= 1:
                continue

            cluster_mems = [chunk[i] for i in group]
            (
                summary_text,
                importance,
                valence,
                intensity,
                trust,
            ) = summarize_cluster(cluster_mems, strategy=strategy)
            if not summary_text:
                continue
            fact_data = factify(summary_text)
            canonical_claim = fact_data.get("canonical_claim")

            existing: list[Memory] = []
            if canonical_claim:
                existing = await store.search(
                    text_query=None,
                    metadata_filters={"canonical_claim": canonical_claim},
                    limit=1,
                )

            if existing:
                existing_mem = existing[0]
                meta = existing_mem.metadata or {}
                src = set(meta.get("source_ids", []))
                src.update(m.id for m in cluster_mems)
                meta["source_ids"] = list(src)
                meta["cluster_size"] = meta.get("cluster_size", 0) + len(cluster_mems)
                meta["trust_score"] = float(np.mean([meta.get("trust_score", trust), trust]))
                if fact_data.get("triples"):
                    meta["triples"] = meta.get("triples", []) + fact_data["triples"]
                if fact_data.get("evidence"):
                    meta["evidence"] = meta.get("evidence", []) + fact_data["evidence"]
                if canonical_claim:
                    meta["canonical_claim"] = canonical_claim
                await store.update_memory(existing_mem.id, metadata=meta)
                for m in cluster_mems:
                    ids_to_remove.append(m.id)
                    modality_map[m.id] = m.modality
                continue

            summary_mem = Memory.new(
                summary_text,
                importance=importance,
                valence=valence,
                emotional_intensity=intensity,
                metadata={
                    "type": "summary",
                    "source_ids": [m.id for m in cluster_mems],
                    "cluster_size": len(cluster_mems),
                    "trust_score": trust,
                    **fact_data,
                },
            )
            meta0 = cluster_mems[0].metadata or {}
            lang = meta0.get("lang") or ""
            source = meta0.get("source") or ""
            normalized = _normalize_for_hash(summary_mem.text)
            key = blake3_hex(f"{normalized}|{lang}|{source}".encode())
            if await event_log.seen("consolidate", key):
                continue
            await event_log.log("consolidate", summary_mem.id, key)
            summary_embedding = embed_text(summary_mem.text)
            if summary_embedding.ndim == 1:
                summary_embedding = np.asarray([summary_embedding], dtype=np.float32)
            pending_adds.append((summary_mem, summary_embedding))
            created.append(summary_mem)
            for m in cluster_mems:
                ids_to_remove.append(m.id)
                modality_map[m.id] = m.modality

    for mem, emb in pending_adds:
        await store.add(mem)
        index.add_vectors([mem.id], emb.astype(np.float32, copy=False))
        MEM_CREATED_TOTAL.labels(mem.modality).inc()
    if created:
        CONSOLIDATIONS_TOTAL.inc(len(created))

    errors: list[str] = []
    if ids_to_remove:
        index.remove_ids(ids_to_remove)
        for mid in ids_to_remove:
            try:
                await store.purge_memory(mid)
                MEM_DELETED_TOTAL.labels(modality_map.get(mid, "text")).inc()
            except RuntimeError as exc:
                log.exception("Failed to delete memory %s during consolidation", mid)
                errors.append(f"delete {mid}: {exc}")

    if errors:
        raise RuntimeError("consolidate_store failures: " + "; ".join(errors))

    return created


@measure_time_async(LAT_FORGET)
async def forget_old_memories(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    min_total: int = 1_000,
    retain_fraction: float = 0.85,
    max_fetch: int = 150_000,
    chunk_size: int = 1_000,
    ttl: float | None = None,
    min_level: int | None = None,
    final: bool | None = None,
    low_trust: float | None = None,
    high_threshold: float = 0.8,
    low_threshold: float = 0.2,
    window_seconds: float = 14 * 86_400,
) -> int:
    """
    Forget the lowest-scoring memories until we drop to the target size.

    Memories with a ``last_accessed`` timestamp older than ``ttl`` seconds are
    always expired: their importance is lowered and they are removed regardless
    of ``retain_fraction``.

    Parameters
    ----------
    min_level:
        If provided, only consider memories with ``level >= min_level``.
    final:
        If set, restrict to memories with matching ``metadata['final']`` value.
        ``False`` also includes memories lacking the flag
    low_trust:
        If set, immediately drop memories with ``trust_score`` at or below this
        threshold before ranking.
    high_threshold:
        Score level that resets the expiry timer when exceeded.
    low_threshold:
        Memories scoring below this may be removed after ``window_seconds`` of
        continuous low score.
    window_seconds:
        Time a memory must remain below ``low_threshold`` before it becomes
        eligible for deletion.

    Returns the number of deleted memories.

    """
    now = _now_utc()
    scored: list[tuple[float, Memory]] = []
    expired: list[Memory] = []
    total = 0
    errors: list[str] = []
    event_log = EventLog(store._dsn)
    deleted_count = 0
    async for chunk in store.search_iter(
        limit=max_fetch,
        chunk_size=chunk_size,
        min_level=min_level,
        final=final,
    ):
        for m in chunk:
            if getattr(m, "pinned", False):
                continue
            last_access = m.created_at
            meta = m.metadata or {}
            ts = meta.get("last_accessed")
            if isinstance(ts, str):
                with contextlib.suppress(ValueError):
                    last_access = dt.datetime.fromisoformat(ts)
            trust = 1.0
            try:
                trust = float(meta.get("trust_score", 1.0))
            except (TypeError, ValueError):
                trust = 1.0
            err = bool(meta.get("error_flag", False))
            has_evidence = bool(meta.get("has_evidence", False))
            high_ts = meta.get("high_score_ts")
            if isinstance(high_ts, str):
                try:
                    high_dt = dt.datetime.fromisoformat(high_ts)
                except ValueError:
                    high_dt = m.created_at
            else:
                high_dt = m.created_at
            if low_trust is not None and trust <= low_trust:
                expired.append(m)
                continue
            if ttl is not None and (now - last_access).total_seconds() > ttl:
                try:
                    await store.update_memory(m.id, importance=0.0)
                except RuntimeError as exc:
                    log.exception("Failed to update memory %s during forget", m.id)
                    errors.append(f"update {m.id}: {exc}")
                expired.append(m)
                continue

            age_days = (now - last_access).total_seconds() / 86_400.0
            score = _decay_score(
                importance=m.importance,
                valence=m.valence,
                emotional_intensity=m.emotional_intensity,
                trust=trust,
                error_flag=err,
                age_days=age_days,
            )
            if score >= high_threshold:
                try:
                    await store.update_memory(m.id, metadata={"high_score_ts": now.isoformat()})
                except RuntimeError as exc:
                    log.exception("Failed to update memory %s during forget", m.id)
                    errors.append(f"update {m.id}: {exc}")
                high_dt = now
            eligible = True
            if score < low_threshold:
                if (now - high_dt).total_seconds() < window_seconds or trust > 0 or has_evidence:
                    eligible = False
            if not eligible:
                continue
            total += 1
            scored.append((score, m))

    if expired:
        ids_to_remove_idx: list[str] = []
        deleted_expired = 0
        for mem in expired:
            meta = mem.metadata or {}
            lang = meta.get("lang") or ""
            source = meta.get("source") or ""
            normalized = _normalize_for_hash(mem.text)
            key = blake3_hex(f"{normalized}|{lang}|{source}".encode())
            if await event_log.seen("forget", key):
                continue
            await event_log.log("forget", mem.id, key)
            ids_to_remove_idx.append(mem.id)
            try:
                await store.purge_memory(mem.id)
                MEM_DELETED_TOTAL.labels(mem.modality).inc()
                deleted_count += 1
                deleted_expired += 1
            except RuntimeError as exc:
                log.exception("Failed to delete memory %s during forget", mem.id)
                errors.append(f"delete {mem.id}: {exc}")
        if ids_to_remove_idx:
            index.remove_ids(ids_to_remove_idx)
        PURGED_TOTAL.inc(deleted_expired)

    if total <= min_total:
        return deleted_count

    keep_count = max(min_total, int(total * retain_fraction))
    forget_count = max(0, total - keep_count)
    ids_to_forget: list[Memory] = []
    if forget_count:
        to_forget = heapq.nsmallest(forget_count, scored, key=lambda x: x[0])
        ids_to_forget = [mem for _, mem in to_forget]

    if ids_to_forget:
        ids_to_remove_idx = []
        for mem in ids_to_forget:
            meta = mem.metadata or {}
            lang = meta.get("lang") or ""
            source = meta.get("source") or ""
            normalized = _normalize_for_hash(mem.text)
            key = blake3_hex(f"{normalized}|{lang}|{source}".encode())
            if await event_log.seen("forget", key):
                continue
            await event_log.log("forget", mem.id, key)
            ids_to_remove_idx.append(mem.id)
            try:
                await store.purge_memory(mem.id)
                MEM_DELETED_TOTAL.labels(mem.modality).inc()
                deleted_count += 1
            except RuntimeError as exc:
                log.exception("Failed to delete memory %s during forget", mem.id)
                errors.append(f"delete {mem.id}: {exc}")
        if ids_to_remove_idx:
            index.remove_ids(ids_to_remove_idx)

    if errors:
        raise RuntimeError("forget_old_memories failures: " + "; ".join(errors))

    return deleted_count


async def periodic_hierarchy_update(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    interval: float = 3_600.0,
    threshold: float = 0.83,
    strategy: str | SummaryStrategy | None = None,
) -> asyncio.Task[None]:
    """
    Start a background task that periodically rebuilds hierarchy levels.

    The task iterates over successive levels starting from ``0`` and builds a
    summary level for each until no further summaries are produced.  It then
    sleeps for ``interval`` seconds before repeating the cycle.  The returned
    :class:`asyncio.Task` can be cancelled by the caller to stop the
    background activity.
    """
    summarizer = HierarchicalSummarizer(
        store,
        index,
        threshold=threshold,
        strategy=strategy,
    )

    async def _worker() -> None:
        while True:
            try:
                level = 0
                while True:
                    created = await summarizer.build_level(level)
                    if not created:
                        break
                    level += 1
            except RuntimeError:
                pass
            await asyncio.sleep(interval)

    return asyncio.create_task(_worker())


async def run_once(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    ttl: float | None = None,
    low_trust: float | None = None,
    high_threshold: float = 0.8,
    low_threshold: float = 0.2,
    window_seconds: float = 14 * 86_400,
) -> None:
    """
    Execute one maintenance cycle.

    Consolidates similar memories and prunes stale ones according to ``ttl``.
    """
    try:
        await consolidate_store(store, index)
    except ModuleNotFoundError:
        # Optional dependencies (numpy/faiss) may be missing in some setups
        log.debug("Skipping consolidation; vector dependencies unavailable")
    await forget_old_memories(
        store,
        index,
        ttl=ttl,
        low_trust=low_trust,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        window_seconds=window_seconds,
    )
    try:
        autotune_index(index)
    except ModuleNotFoundError:
        log.debug("Skipping autotune; vector dependencies unavailable")


async def maintenance_loop(
    interval: float = 3_600.0,
    ttl: float | None = None,
    low_trust: float | None = None,
    high_threshold: float | None = None,
    low_threshold: float | None = None,
    window_seconds: float | None = None,
) -> None:
    """Run maintenance forever with a sleep of ``interval`` seconds."""
    settings = UnifiedSettings()
    async with EnhancedMemoryStore(settings) as enhanced:
        store = enhanced._store
        index = cast(
            "FaissHNSWIndex",
            getattr(enhanced.vector_store, "index", enhanced.vector_store),
        )
        if high_threshold is None:
            high_threshold = settings.maintenance.forget_high_threshold
        if low_threshold is None:
            low_threshold = settings.maintenance.forget_low_threshold
        if window_seconds is None:
            window_seconds = settings.maintenance.forget_window_days * 86_400
        while True:
            try:
                await run_once(
                    store,
                    index,
                    ttl=ttl,
                    low_trust=low_trust,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                    window_seconds=window_seconds,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.exception("maintenance iteration failed: %s", exc)
            await asyncio.sleep(interval)
