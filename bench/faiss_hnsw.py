"""
Simple FAISS HNSW benchmark with deterministic data.

This module builds an HNSW index, inserts a fixed set of vectors and
performs a deterministic set of queries.  Both the insertion and search
use pseudo-random data generated from a fixed seed so that benchmark
results are reproducible across runs.
"""

from __future__ import annotations

import logging
import time

import numpy.random as npr

logger = logging.getLogger(__name__)


class BenchError(RuntimeError):
    """Error raised for benchmark configuration issues."""


try:  # pragma: no cover - optional dependency
    import faiss
except Exception as exc:  # pragma: no cover - make import error explicit
    message = "faiss is required for the FAISS HNSW benchmark"
    raise BenchError(message) from exc


def run_benchmark(
    n_vectors: int = 1000,
    dim: int = 64,
    n_queries: int = 10,
    k: int = 5,
    seed: int = 1234,
) -> tuple[float, float]:
    """
    Build an HNSW index and measure insertion and search times.

    Parameters
    ----------
    n_vectors:
        Number of vectors to insert into the index.
    dim:
        Dimensionality of each vector.
    n_queries:
        How many search queries to execute.
    k:
        Number of nearest neighbours to retrieve per query.
    seed:
        Seed for the pseudo-random number generator.

    Returns
    -------
    Tuple[float, float]
        A tuple ``(build_time, search_time)`` in seconds.

    """
    rng = npr.default_rng(seed)
    vectors = rng.random((n_vectors, dim), dtype="float32")

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64

    start = time.perf_counter_ns()
    index.add(vectors)
    build_time = (time.perf_counter_ns() - start) / 1e9

    queries = rng.random((n_queries, dim), dtype="float32")

    start = time.perf_counter_ns()
    for q in queries:
        index.search(q.reshape(1, -1), k)
    search_time = (time.perf_counter_ns() - start) / 1e9

    return build_time, search_time


def main() -> None:  # pragma: no cover - convenience entry point
    """Run the benchmark and log the results."""
    build, search = run_benchmark()
    logger.info("Build time: %.6fs\nSearch time for 10 queries: %.6fs", build, search)


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
