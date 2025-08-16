"""Regression benchmark for the FAISS HNSW index."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.needs_faiss

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from bench.faiss_hnsw import run_benchmark

# Baseline timings in milliseconds with generous tolerance.
BASE_BUILD_MS = 100.0
BASE_SEARCH_MS = 20.0  # total for 10 queries
TOLERANCE = 0.5  # 50% tolerance

MAX_BUILD_MS = float(os.getenv("MAX_HNSW_BUILD_MS", str(BASE_BUILD_MS * (1 + TOLERANCE))))
MAX_SEARCH_MS = float(os.getenv("MAX_HNSW_SEARCH_MS", str(BASE_SEARCH_MS * (1 + TOLERANCE))))


@pytest.mark.perf
def test_hnsw_build_and_search_speed() -> None:
    build, search = run_benchmark()
    build_ms = build * 1000
    search_ms = search * 1000
    assert build_ms < MAX_BUILD_MS, f"index build took {build_ms:.2f} ms (limit {MAX_BUILD_MS} ms)"
    assert search_ms < MAX_SEARCH_MS, f"search took {search_ms:.2f} ms (limit {MAX_SEARCH_MS} ms)"
