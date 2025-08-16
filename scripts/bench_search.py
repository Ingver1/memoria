from __future__ import annotations

"""Benchmark search performance for vector vs. hybrid retrieval."""

import argparse
import asyncio
import gc
import random
import secrets
import statistics
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - make import error explicit

    class _NumpyStub:
        __stub__ = True
        float32 = float

        class ndarray(list):
            pass

    np = _NumpyStub()

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover - make import error explicit
    faiss = None

from memory_system.unified_memory import Memory, SearchResults, _get_cache, search

DIM = 64
TOP_K = 5
RERANK_K = 20
MMR_LAMBDA = 0.7


class DummyStore:
    def __init__(self, memories: list[Memory]):
        self.memories = memories

    async def search_memory(
        self,
        *,
        query: str,
        k: int = 5,
        metadata_filter=None,
        level=None,
        context=None,
    ) -> SearchResults:
        tokens = set(query.split())
        scored: list[tuple[int, Memory]] = []
        for m in self.memories:
            score = sum(1 for tok in tokens if tok in m.text)
            if score:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in scored[:k]]
        shadow = [m for m in results if (m.metadata or {}).get("shadow")]
        primary = [m for m in results if (m.metadata or {}).get("shadow") is not True]
        return SearchResults(primary, shadow)

    async def semantic_search(
        self,
        *,
        embedding: Iterable[float],
        k: int = 5,
        return_distance: bool = False,
        mmr_lambda=None,
        metadata_filter=None,
        level=None,
        context=None,
        modality="text",
    ):
        emb = list(embedding)
        scored: list[tuple[Memory, float]] = []
        for m in self.memories:
            dot = sum(a * b for a, b in zip(emb, m._embedding, strict=False))
            norm_a = sum(a * a for a in emb) ** 0.5 or 1.0
            norm_b = sum(b * b for b in m._embedding) ** 0.5 or 1.0
            cos_sim = dot / (norm_a * norm_b)
            cos_dist = 1.0 - cos_sim
            scored.append((m, cos_dist))
        scored.sort(key=lambda x: x[1])
        top = scored[:k]
        if return_distance:
            return top
        return [m for m, _ in top]


class _FakeVec(list):
    def tolist(self):
        return list(self)


def fake_embed(text: str) -> _FakeVec:
    random.seed(hash(text) % (2**32))
    return _FakeVec(secrets.randbelow(1000) / 1000 for _ in range(DIM))


def make_memories(n: int) -> list[Memory]:
    now = datetime.now(UTC)
    memories: list[Memory] = []
    for i in range(n):
        text = f"memory {i}"
        m = Memory(str(i), text, now)
        m._embedding = [secrets.randbelow(1000) / 1000 for _ in range(DIM)]
        memories.append(m)
    return memories


async def search_once(store, retriever: str, use_mmr: bool, use_cross: bool) -> None:
    kwargs = {"retriever": retriever, "k": RERANK_K if use_cross else TOP_K, "store": store}
    if use_mmr:
        kwargs["mmr_lambda"] = MMR_LAMBDA
    if use_cross:
        kwargs.update({"reranker": "cross", "k_vec": RERANK_K, "k_bm25": RERANK_K})
        from types import ModuleType, SimpleNamespace

        settings_mod = ModuleType("settings")
        settings_mod.get_settings = lambda env=None: SimpleNamespace(
            ranking=SimpleNamespace(use_cross_encoder=True, alpha=1.0, beta=1.0, gamma=1.0)
        )
        sys.modules["memory_system.settings"] = settings_mod

        fake_cr = ModuleType("cross_reranker")

        def fake_order(mems, query, **_):
            out = []
            for m in mems:
                meta = dict(m.metadata or {})
                meta["cross_score"] = 1.0
                m.metadata = meta
                out.append(m)
            return out

        fake_cr.order_by_cross_encoder = fake_order
        sys.modules["memory_system.core.cross_reranker"] = fake_cr
    await search("memory", **kwargs)


async def bench(fn, reps: int = 20, warmup: int = 5) -> dict[str, float]:
    gc.disable()
    for _ in range(warmup):
        await fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        await fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    gc.enable()
    median = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18]
    qps = 1000.0 / median if median else 0.0
    return {"median_ms": median, "p95_ms": p95, "qps": qps}


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


def tune_faiss(dim: int = DIM, samples: int = 1000) -> None:
    if faiss is None or getattr(np, "__stub__", False):
        raise RuntimeError("faiss and numpy are required for tuning")
    rng = np.random.default_rng(0)
    vecs = rng.random((samples, dim), dtype=np.float32)
    flat = faiss.IndexFlatL2(dim)
    flat.add(vecs)
    k = min(5, samples)
    _, gt = flat.search(vecs, k)
    results: list[tuple[float, float, int, int]] = []
    for M in (16, 32, 64):
        for ef in (50, 100, 200):
            idx = faiss.IndexHNSWFlat(dim, M)
            idx.hnsw.efConstruction = max(ef * 2, 100)
            idx.hnsw.efSearch = ef
            idx.add(vecs)
            start = time.perf_counter_ns()
            _, I = idx.search(vecs, k)
            lat = (time.perf_counter_ns() - start) / 1e6 / samples
            recall = 0.0
            for q in range(samples):
                recall += len(set(I[q]) & set(gt[q])) / k
            recall /= samples
            results.append((recall, lat, M, ef))
    results.sort(key=lambda x: (-x[0], x[1]))
    best_recall, best_lat, best_M, best_ef = results[0]
    best_nprobe = best_ef
    _update_env_file(
        {
            "AI_FAISS__M": best_M,
            "AI_FAISS__EF_SEARCH": best_ef,
            "AI_FAISS__NPROBE": best_nprobe,
        }
    )
    print(
        f"Selected M={best_M} efSearch={best_ef} nprobe={best_nprobe} "
        f"recall={best_recall:.3f} latency={best_lat:.2f}ms"
    )


async def main() -> None:
    random.seed(42)
    for n in (1000, 10000):
        memories = make_memories(n)
        store = DummyStore(memories)

        from types import ModuleType

        fake_embedder = ModuleType("embedder")
        fake_embedder.embed = fake_embed
        sys.modules["embedder"] = fake_embedder

        print(f"\nDataset size: {n}")
        combos = [
            ("vector", False, False),
            ("vector", True, True),
            ("hybrid", False, False),
            ("hybrid", True, True),
        ]
        for retriever, use_mmr, use_cross in combos:
            _get_cache().clear()
            stats = await bench(
                lambda store=store,
                retriever=retriever,
                use_mmr=use_mmr,
                use_cross=use_cross: search_once(store, retriever, use_mmr, use_cross)
            )
            label = retriever
            if use_mmr:
                label += f"+mmr(λ={MMR_LAMBDA})"
            if use_cross:
                label += "+cross"
            print(
                f"{label:22s} median={stats['median_ms']:.2f}ms p95={stats['p95_ms']:.2f}ms qps={stats['qps']:.1f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark search or tune FAISS parameters")
    parser.add_argument("--tune", action="store_true", help="Autotune FAISS parameters")
    args = parser.parse_args()
    if args.tune:
        tune_faiss()
    else:
        asyncio.run(main())
