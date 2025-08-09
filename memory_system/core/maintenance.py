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
import datetime as dt
from typing import Dict, List, Sequence, Tuple

import numpy as np

from embedder import embed as embed_text
from memory_system.core.hierarchical_summarizer import HierarchicalSummarizer
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.summarization import STRATEGIES, SummaryStrategy

# ----------------------------- small utils -----------------------------


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1D embeddings (assumed L2-normalized)."""
    return float(np.dot(a, b))


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ----------------------------- clustering -----------------------------


def cluster_memories(
    embeddings: Sequence[np.ndarray],
    threshold: float = 0.83,
) -> List[List[int]]:
    """
    Greedy single-pass clustering by cosine similarity to the first item in
    each cluster. O(n²) in the worst case but simple and robust for small/medium
    collections.

    Returns a list of clusters; each cluster is a list of indices into
    `embeddings`.
    """
    n = len(embeddings)
    visited = [False] * n
    clusters: List[List[int]] = []

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


def summarize_cluster(
    memories: Sequence[Memory],
    *,
    strategy: str | SummaryStrategy = "head2tail",
) -> Tuple[str, float, float, float]:
    """Summarise a cluster of memories and compute aggregated attributes.

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
        return "", 0.0, 0.0, 0.0

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
    return summary_text, avg_importance, avg_valence, avg_intensity


# ------------------------ consolidation / forgetting -------------------


async def consolidate_store(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    threshold: float = 0.83,
    max_fetch: int = 100_000,
    strategy: str | SummaryStrategy = "head2tail",
) -> List[Memory]:
    """
    Cluster similar memories, create a summary memory per cluster, then delete
    the originals that were consolidated.

    Returns the list of created summary memories.
    """
    # Fetch a large batch; the store does not have "fetch all" semantics.
    all_mems = await store.search(limit=max_fetch)
    if not all_mems:
        return []

    # Batch embeddings (faster than per-item).
    texts = [m.text for m in all_mems]
    embeddings = embed_text(texts)  # shape: (N, D)
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    assert embeddings.shape[0] == len(all_mems), "Embedding batch size mismatch"

    clusters = cluster_memories([embeddings[i] for i in range(embeddings.shape[0])], threshold)

    created: List[Memory] = []
    ids_to_remove: List[str] = []

    for group in clusters:
        # Ignore singletons (no consolidation needed).
        if len(group) <= 1:
            continue

        cluster_mems = [all_mems[i] for i in group]
        summary_text, importance, valence, intensity = summarize_cluster(
            cluster_mems, strategy=strategy
        )
        if not summary_text:
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
            },
        )

        # Persist new memory & add embedding to ANN index.
        await store.add(summary_mem)
        summary_embedding = embed_text(summary_text)
        if summary_embedding.ndim == 1:
            summary_embedding = np.asarray([summary_embedding], dtype=np.float32)
        index.add_vectors([summary_mem.id], summary_embedding.astype(np.float32, copy=False))
        created.append(summary_mem)

        # Schedule originals for removal.
        ids_to_remove.extend(m.id for m in cluster_mems)

    if ids_to_remove:
        # Remove embeddings first, then rows.
        index.remove_ids(ids_to_remove)
        for mid in ids_to_remove:
            try:
                await store.delete_memory(mid)
            except Exception:
                # If a row is already gone, keep going.
                pass

    return created


def _decay_score(
    *,
    importance: float,
    valence: float,
    emotional_intensity: float,
    age_days: float,
) -> float:
    """
    Age-aware retention score (higher -> keep):

        base = 0.5*importance + 0.3*emotional_intensity + 0.2*max(valence, 0)
        score = base * exp(-age_days / 30)

    Tuned to decay over ~1 month while preserving high-importance/intense items.
    """
    base = 0.5 * importance + 0.3 * emotional_intensity + 0.2 * max(valence, 0.0)
    decay = float(np.exp(-age_days / 30.0))
    return base * decay


async def forget_old_memories(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    min_total: int = 1_000,
    retain_fraction: float = 0.85,
    max_fetch: int = 150_000,
) -> int:
    """
    Forget the lowest-scoring memories until we drop to the target size.

    Returns the number of deleted memories.
    """
    all_mems = await store.search(limit=max_fetch)
    total = len(all_mems)
    if total <= min_total:
        return 0

    now = _now_utc()

    # Compute scores.
    scores: Dict[str, float] = {}
    for m in all_mems:
        age_days = max(0.0, (now - m.created_at).total_seconds() / 86_400.0)
        scores[m.id] = _decay_score(
            importance=m.importance,
            valence=m.valence,
            emotional_intensity=m.emotional_intensity,
            age_days=age_days,
        )

    # Keep the top-K by score; forget the rest.
    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    keep_count = max(min_total, int(total * retain_fraction))
    ids_to_forget = sorted_ids[keep_count:]

    if ids_to_forget:
        index.remove_ids(ids_to_forget)
        for mid in ids_to_forget:
            try:
                await store.delete_memory(mid)
            except Exception:
                pass

    return len(ids_to_forget)


async def periodic_hierarchy_update(
    store: SQLiteMemoryStore,
    index: FaissHNSWIndex,
    *,
    interval: float = 3_600.0,
    threshold: float = 0.83,
    strategy: str | SummaryStrategy = "head2tail",
) -> asyncio.Task:
    """Start a background task that periodically rebuilds hierarchy levels.

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
            except Exception:
                pass
            await asyncio.sleep(interval)

    return asyncio.create_task(_worker())
