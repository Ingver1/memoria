"""Tests for FAISS GPU detection and fallback."""

import logging
from types import SimpleNamespace

import pytest

from memory_system.core import index as idx_module


class _IndexHNSWFlat:
    def __init__(self, dim: int, m: int, metric: int) -> None:
        self.hnsw = SimpleNamespace(efConstruction=0, efSearch=0)


class _IndexIDMap2:
    def __init__(self, base: object) -> None:
        self.index = base


def _make_faiss_cpu() -> SimpleNamespace:
    return SimpleNamespace(
        METRIC_INNER_PRODUCT=0,
        METRIC_L2=1,
        IndexHNSWFlat=_IndexHNSWFlat,
        IndexIDMap2=_IndexIDMap2,
    )


def _make_faiss_gpu() -> SimpleNamespace:
    ns = _make_faiss_cpu()
    ns.called = False

    def get_num_gpus() -> int:  # pragma: no cover - simple
        return 1

    def index_cpu_to_all_gpus(base: object) -> object:  # pragma: no cover - simple
        ns.called = True
        return base

    ns.get_num_gpus = get_num_gpus
    ns.index_cpu_to_all_gpus = index_cpu_to_all_gpus
    return ns


def test_force_cpu_when_gpu_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ns = _make_faiss_cpu()
    monkeypatch.setattr(idx_module, "faiss", ns)
    caplog.set_level(logging.WARNING)
    idx = idx_module.FaissHNSWIndex(3, use_gpu=True)
    assert idx.use_gpu is False
    assert "faiss-gpu" in caplog.text


def test_gpu_path_when_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ns = _make_faiss_gpu()
    monkeypatch.setattr(idx_module, "faiss", ns)
    caplog.set_level(logging.INFO)
    idx = idx_module.FaissHNSWIndex(3, use_gpu=True)
    assert idx.use_gpu is True
    assert ns.called is True
    assert "faiss-gpu" not in caplog.text
