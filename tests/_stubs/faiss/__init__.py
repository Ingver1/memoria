"""Minimal faiss stub for environments without the real library.

Provides the handful of classes and constants used by the tests and the
FAISS wrapper implementation. It is intentionally tiny and behaviour is
only loosely approximated.
"""

from __future__ import annotations


METRIC_L2 = 1
METRIC_INNER_PRODUCT = 2


class Index:
    def __init__(self, dim: int) -> None:  # pragma: no cover - trivial
        self.dim = dim


class _HNSW:
    def __init__(self) -> None:  # pragma: no cover - trivial
        self.efConstruction = 0
        self.efSearch = 0


class IndexHNSWFlat(Index):
    def __init__(self, dim: int, M: int, metric: int) -> None:  # pragma: no cover - trivial
        super().__init__(dim)
        self.M = M
        self.metric = metric
        self.hnsw = _HNSW()


class IndexFlatL2(Index):  # pragma: no cover - simple stub
    pass


class IndexFlatIP(Index):  # pragma: no cover - simple stub
    pass


class IndexIVFFlat(Index):  # pragma: no cover - simple stub
    def __init__(self, quantizer: Index, dim: int, nlist: int, metric: int) -> None:
        super().__init__(dim)
        self.quantizer = quantizer
        self.nlist = nlist
        self.metric = metric
        self.nprobe = 1


class IndexIVFPQ(Index):  # pragma: no cover - simple stub
    def __init__(self, quantizer: Index, dim: int, nlist: int, m: int, bits: int, metric: int) -> None:
        super().__init__(dim)
        self.quantizer = quantizer
        self.nlist = nlist
        self.m = m
        self.bits = bits
        self.metric = metric
        self.nprobe = 1


class OPQMatrix:  # pragma: no cover - simple stub
    def __init__(self, dim: int, m: int) -> None:
        self.dim = dim
        self.m = m


class IndexPreTransform(Index):  # pragma: no cover - simple stub
    def __init__(self, opq: OPQMatrix, base: Index) -> None:
        super().__init__(base.dim)
        self.base = base


class IndexIDMap2(Index):  # pragma: no cover - simple stub
    def __init__(self, base: Index) -> None:
        super().__init__(base.dim)
        self.base = base


def index_cpu_to_all_gpus(index: Index) -> Index:  # pragma: no cover - simple stub
    return index


def read_index(path: str) -> Index:  # pragma: no cover
    return Index()


def write_index(index: Index, path: str) -> None:  # pragma: no cover
    return None


def normalize_L2(arr):  # pragma: no cover - simple stub
    # No-op normalisation for tests
    return None
