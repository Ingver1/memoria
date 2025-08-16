"""Tests for kNN-LM mixing utilities."""

import time

import pytest

from memory_system.utils.knn_lm import NeighborCache, knn_lm_mix


def test_knn_lm_mix_basic() -> None:
    probs_lm = [0.2, 0.8]
    probs_knn = [1.0, 0.0]
    mixed = knn_lm_mix(probs_lm, probs_knn, 0.5)
    assert mixed == pytest.approx([0.6, 0.4])


def test_knn_lm_mix_fallback() -> None:
    probs_lm = [0.1, 0.9]
    mixed = knn_lm_mix(probs_lm, None, 0.7)
    assert mixed == pytest.approx(probs_lm)


def test_knn_lm_mix_limit() -> None:
    probs_lm = [0.9, 0.1]
    probs_knn = [0.0, 1.0]
    mixed = knn_lm_mix(probs_lm, probs_knn, 0.5, max_knn_ratio=0.4)
    assert mixed == pytest.approx(probs_lm)


def test_knn_lm_mix_forbidden() -> None:
    probs_lm = [0.9, 0.1]
    probs_knn = [0.0, 1.0]
    mixed = knn_lm_mix(probs_lm, probs_knn, 0.5, forbidden_tokens=[1])
    assert mixed == pytest.approx(probs_lm)


def test_neighbor_cache() -> None:
    class DummyIndex:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query, k: int = 1):
            self.calls += 1
            return [(1, 0.0)]

    index = DummyIndex()
    cache = NeighborCache(index, vocab_size=2)
    query = [0.0]
    first = cache.get_probabilities(query)
    second = cache.get_probabilities(query)
    assert index.calls == 1
    assert first == pytest.approx(second)


def test_neighbor_cache_empty() -> None:
    class EmptyIndex:
        def search(self, query, k: int = 1):
            return []

    cache = NeighborCache(EmptyIndex(), vocab_size=2)
    assert cache.get_probabilities([0.0]) is None


def test_neighbor_cache_ttl() -> None:
    class DummyIndex:
        def __init__(self) -> None:
            self.calls = 0

        def search(self, query, k: int = 1):
            self.calls += 1
            return [(0, 0.0)]

    index = DummyIndex()
    cache = NeighborCache(index, vocab_size=1, ttl=0.01)
    q = [0.0]
    cache.get_probabilities(q)
    time.sleep(0.02)
    cache.get_probabilities(q)
    assert index.calls == 2
