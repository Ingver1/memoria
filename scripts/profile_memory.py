"""
Profile summarization and memory dynamics routines.

This script uses ``cProfile`` to measure hotspots in clustering,
summary generation and scoring utilities.  It generates synthetic
memories and embeddings so it can run without external services.
"""

from __future__ import annotations

import cProfile
import pstats
from random import random

import numpy as np
from numpy.typing import NDArray

from memory_system.core.maintenance import cluster_memories, summarize_cluster
from memory_system.core.memory_dynamics import MemoryDynamics
from memory_system.core.store import Memory


def _sample_embeddings(n: int) -> list[NDArray[np.float32]]:
    """Return ``n`` normalised random vectors for clustering."""
    vecs = [np.random.rand(384).astype(np.float32) for _ in range(n)]
    for v in vecs:
        v /= np.linalg.norm(v)
    return vecs


def profile_summarization() -> None:
    """Run clustering and summarisation on synthetic data."""
    memories = [
        Memory.new(f"memory-{i}", importance=random(), emotional_intensity=random())
        for i in range(200)
    ]
    embeddings = _sample_embeddings(len(memories))
    clusters = cluster_memories(embeddings)
    for cluster in clusters:
        summarize_cluster([memories[i] for i in cluster])


def profile_dynamics() -> None:
    """Exercise the scoring routine of :class:`MemoryDynamics`."""
    dyn = MemoryDynamics()
    mem = Memory.new("probe", importance=0.5, emotional_intensity=0.5)
    for _ in range(1000):
        dyn.score(mem)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    profile_summarization()
    profile_dynamics()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
