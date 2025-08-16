"""
Hierarchical summarisation utilities.

This module implements a simple RAPTOR-like approach for building a
hierarchy of clustered memories. Memories have an integer ``level`` where
``0`` denotes raw inputs. Memories on a given level are grouped by cosine
similarity; each cluster is summarised into a new memory on the next
level. Items without sufficiently similar neighbours are marked with
``metadata['final']`` so they are excluded from future promotion. The
summarised memories are written back to the
:class:`~memory_system.core.store.SQLiteMemoryStore` and added to the
vector index so they can participate in subsequent searches.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from embedder import embed as embed_text
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.summarization import STRATEGIES, SummaryStrategy


def _cos_sim(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity for two 1-D L2-normalised vectors."""
    return float(np.dot(a, b))


def _cluster_embeddings(
    embeddings: Sequence[NDArray[np.float32]], threshold: float
) -> list[list[int]]:
    """
    Greedy single-pass clustering.

    Returns a list of clusters where each cluster is a list of indices into
    ``embeddings``.  Two items belong to the same cluster when their cosine
    similarity is above ``threshold``.
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


class HierarchicalSummarizer:
    """
    Build and maintain summary levels for stored memories.

    Each pass groups memories from a source ``level`` and writes a summary
    to ``level + 1``. Memories that cannot be clustered are tagged with
    ``metadata['final']`` and will not be considered in subsequent passes.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        index: FaissHNSWIndex,
        *,
        threshold: float = 0.83,
        strategy: str | SummaryStrategy | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.threshold = threshold
        if strategy is None:
            try:
                from memory_system.settings import get_settings

                cfg = get_settings()
                strategy = getattr(cfg, "summary_strategy", "head2tail")
            except ImportError:  # pragma: no cover - settings module optional
                strategy = "head2tail"
        if isinstance(strategy, str):
            self.strategy = STRATEGIES[strategy]
        else:
            self.strategy = strategy

    async def build_level(self, source_level: int) -> list[Memory]:
        """
        Build the next summary level from ``source_level``.

        Memories already marked ``metadata['final']`` are skipped. Returns a
        list of newly created summary memories belonging to ``source_level +
        1``. Singletons are marked as final to prevent further promotion. If
        no memories exist on the source level the function returns an empty
        list.
        """
        mems = await self.store.search(limit=100_000, level=source_level)
        mems = [m for m in mems if not m.metadata or not m.metadata.get("final")]
        if not mems:
            return []

        texts = [m.text for m in mems]
        embeddings = embed_text(texts)
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] != len(mems):
            raise ValueError(
                "Number of embeddings does not match number of memories: "
                f"{embeddings.shape[0]} != {len(mems)}"
            )

        clusters = _cluster_embeddings(
            [embeddings[i] for i in range(embeddings.shape[0])], self.threshold
        )

        created: list[Memory] = []
        target_level = source_level + 1
        for group in clusters:
            if len(group) <= 1:
                # Mark singletons as final to prevent further promotion.
                m = mems[group[0]]
                await self.store.update_memory(m.id, metadata={"final": True})
                continue

            cluster_mems = [mems[i] for i in group]
            summary_text = self.strategy(cluster_mems)
            importance = float(np.mean([m.importance for m in cluster_mems]))
            valence = float(np.mean([m.valence for m in cluster_mems]))
            intensity = float(np.mean([m.emotional_intensity for m in cluster_mems]))
            trust = float(
                np.mean([(m.metadata or {}).get("trust_score", 0.0) for m in cluster_mems])
            )
            summary = Memory.new(
                summary_text,
                importance=importance,
                valence=valence,
                emotional_intensity=intensity,
                metadata={
                    "source_ids": [m.id for m in cluster_mems],
                    "cluster_size": len(cluster_mems),
                    "trust_score": trust,
                },
                level=target_level,
            )
            await self.store.add(summary)
            summary_vec = embed_text(summary_text)
            if summary_vec.ndim == 1:
                summary_vec = np.asarray([summary_vec], dtype=np.float32)
            self.index.add_vectors([summary.id], summary_vec.astype(np.float32, copy=False))
            created.append(summary)
        return createdd
