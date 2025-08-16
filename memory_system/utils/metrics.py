"""
Unified Memory System — Prometheus metrics utilities.

This module defines a common bucket configuration for latency histograms
so that dashboards can derive latency percentiles using
``histogram_quantile()``. A typical PromQL query for the 95th percentile
latency over the last five minutes is::

    histogram_quantile(0.95, rate(ums_db_query_latency_seconds_bucket[5m]))

"""

from __future__ import annotations

from memory_system import __version__

# Update module docstring with current version
__doc__ = __doc__.replace("utilities", f"utilities (v{__version__})")

import logging
import os
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ContextManager, ParamSpec, Protocol, Self, TypeVar, cast

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class Metric(Protocol):
    def labels(self, *args: Any, **kwargs: Any) -> Metric: ...
    def inc(self, amount: float = 1.0) -> None: ...
    def observe(self, value: float) -> None: ...
    def set(self, value: float) -> None: ...
    def time(self) -> ContextManager[Any]: ...


if TYPE_CHECKING:
    from collections.abc import Callable

    from prometheus_client import CollectorRegistry as CollectorRegistryT

    CounterT = Callable[..., Metric]
    GaugeT = Callable[..., Metric]
    HistogramT = Callable[..., Metric]
    generate_latest_t = Callable[[CollectorRegistryT | None], bytes]
    make_asgi_app_t = Callable[[CollectorRegistryT | None], ASGIApp]

_pc: ModuleType | None
try:  # pragma: no cover - optional dependency
    import prometheus_client as _pc
except Exception:  # pragma: no cover - fallback when lib missing
    _pc = None

if _pc is not None:
    CONTENT_TYPE_LATEST = _pc.CONTENT_TYPE_LATEST
    CollectorRegistry = cast("Any", _pc.CollectorRegistry)
    Counter = cast("CounterT", _pc.Counter)
    Gauge = cast("GaugeT", _pc.Gauge)
    Histogram = cast("HistogramT", _pc.Histogram)
    generate_latest = cast("generate_latest_t", _pc.generate_latest)
    make_asgi_app = cast("make_asgi_app_t", _pc.make_asgi_app)
    REGISTRY = cast("Any", _pc.REGISTRY)
else:

    class _MetricBase:
        """Lightweight metric with the Prometheus client API."""

        def labels(self, *args: Any, **kwargs: Any) -> Self:
            return self

        def inc(self, amount: float = 1.0) -> None:
            self._value += amount

        def observe(self, value: float) -> None:
            self._value += value

        def set(self, value: float) -> None:
            self._value = value

        def time(self) -> ContextManager[Any]:
            return nullcontext()

    class _Counter(_MetricBase):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value = 0.0

    class _Gauge(_MetricBase):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value = 0.0

    class _Histogram(_MetricBase):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._value = 0.0
            self._samples: list[float] = []

        def observe(self, value: float) -> None:
            self._samples.append(value)

    class _CollectorRegistry:
        def __init__(self) -> None:
            self._metrics: list[_MetricBase] = []

        def register(self, metric: _MetricBase) -> None:
            self._metrics.append(metric)

        def collect(self) -> list[_MetricBase]:
            return list(self._metrics)

    Counter = _Counter
    Gauge = _Gauge
    Histogram = _Histogram
    CollectorRegistry = _CollectorRegistry
    REGISTRY = CollectorRegistry()

    def make_asgi_app(registry: CollectorRegistryT | None = None) -> ASGIApp:
        async def _app(scope: Scope, receive: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        return _app

    CONTENT_TYPE_LATEST = "text/plain"

    def generate_latest(registry: CollectorRegistryT | None = None) -> bytes:
        return b""


# Registry -----------------------------------------------------------------

log = logging.getLogger(__name__)


def _build_registry() -> CollectorRegistryT:
    """Return a ``CollectorRegistry`` with optional multiprocess support."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return cast("CollectorRegistryT", REGISTRY)

    registry = cast(
        "CollectorRegistryT", CollectorRegistry()
    )  # fresh registry for multiprocess mode
    try:  # pragma: no cover - best effort cleanup
        for file in Path(multiproc_dir).glob("*.db"):
            file.unlink(missing_ok=True)
    except Exception:  # pylint: disable=broad-except
        log.debug("could not clean multiprocess metrics", exc_info=True)

    try:  # pragma: no cover - optional multiprocess collector
        from prometheus_client import multiprocess

        cast("Any", multiprocess.MultiProcessCollector)(registry)
    except Exception:  # pylint: disable=broad-except
        log.debug("multiprocess collector unavailable", exc_info=True)

    return registry


REGISTRY = _build_registry()
metrics_app = make_asgi_app(REGISTRY)

# Buckets in seconds for latency histograms. These buckets cover typical
# operations from a few milliseconds up to several seconds and are suitable
# for use with ``histogram_quantile()``.
LATENCY_BUCKETS = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]


# Helper for creating counters (not used for fixed metrics below, but available for dynamic creation)
def prometheus_counter(name: str, description: str, labels: list[str] | None = None) -> Metric:
    """Create a Prometheus Counter with optional label names."""
    if labels:
        return Counter(name, description, tuple(labels))
    return Counter(name, description)


# Base metrics collectors
MET_ERRORS_TOTAL = Counter("ums_errors_total", "Total errors", ("type", "component"))
LAT_DB_QUERY = Histogram(
    "ums_db_query_latency_seconds", "DB query latency", buckets=LATENCY_BUCKETS
)
LAT_SEARCH = Histogram(
    "ums_search_latency_seconds", "Vector search latency", buckets=LATENCY_BUCKETS
)
LAT_EMBEDDING = Histogram(
    "ums_embedding_latency_seconds", "Embedding generation latency", buckets=LATENCY_BUCKETS
)
MET_POOL_EXHAUSTED = Counter("ums_pool_exhausted_total", "Connection pool exhausted events")

# Maintenance metrics
LAT_CONSOLIDATION = Histogram(
    "ums_consolidation_latency_seconds",
    "Consolidation operation latency",
    buckets=LATENCY_BUCKETS,
)
LAT_FORGET = Histogram(
    "ums_forget_latency_seconds", "Forgetting operation latency", buckets=LATENCY_BUCKETS
)
MEM_CREATED_TOTAL = Counter("ums_memories_created_total", "Memories created", ("modality",))
MEM_DELETED_TOTAL = Counter("ums_memories_deleted_total", "Memories deleted", ("modality",))
PURGED_TOTAL = Counter("ums_purged_total", "Memories purged due to expiration")
CONSOLIDATIONS_TOTAL = Counter("ums_consolidations_total", "Memory consolidations performed")
DRIFT_COUNT = Counter("ums_index_drift_total", "Vector index drift detections")
HEAL_COUNT = Counter("ums_index_heal_total", "Memories reindexed or removed to heal drift")

# Recall monitoring
RECALL_AT_K = Gauge("ums_recall_at_k", "Average recall@k across control queries")

# Memory quality metrics
MEMORY_ACCEPT_RATE = Gauge("ums_memory_accept_rate", "Ratio of memories accepted by quality gate")
LESSON_PROMOTION_RATE = Gauge("ums_lesson_promotion_rate", "Rate at which lessons are promoted")
RETRIEVAL_HIT_AT_K = Gauge("ums_retrieval_hit_at_k", "Hit rate of retrievals at k")
QUALITY_GATE_DROP_REASON = Counter(
    "ums_quality_gate_drop_reason_total",
    "Memories dropped by quality gate",
    ("reason",),
)
CONTRIB_SCORE_HIST = Histogram(
    "ums_contrib_score",
    "Contribution score distribution",
    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
)

# Cache and queue metrics
CACHE_HITS_TOTAL = Counter("ums_cache_hits_total", "Cache hits")
CACHE_MISSES_TOTAL = Counter("ums_cache_misses_total", "Cache misses")
CACHE_HIT_RATE = Gauge("ums_cache_hit_rate", "Cache hit rate")
EMBEDDING_QUEUE_LENGTH = Gauge("ums_embedding_queue_length", "Embedding queue length")
LAT_EMBEDDING_WAIT = Histogram(
    "ums_embedding_queue_wait_seconds",
    "Time spent waiting in embedding queue",
    buckets=LATENCY_BUCKETS,
)

# System metrics gauges
SYSTEM_CPU = Gauge("ums_system_cpu_percent", "CPU usage percentage")
SYSTEM_MEM = Gauge("ums_system_mem_percent", "Memory usage percentage")
PROCESS_UPTIME = Gauge("ums_process_uptime_seconds", "Process uptime in seconds")

_START_TIME = time.monotonic()
PROCESS_UPTIME.set(0.0)

# Timing decorators for measuring execution time of functions


P = ParamSpec("P")
R = TypeVar("R")


def _wrap_sync(metric: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory for synchronous function timing."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            timer = getattr(metric, "time", nullcontext)
            with timer():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _wrap_async(
    metric: Any,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator factory for async function timing."""

    def decorator(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            timer = getattr(metric, "time", nullcontext)
            with timer():
                return await fn(*args, **kwargs)

        return wrapper

    return decorator


measure_time = _wrap_sync
measure_time_async = _wrap_async

# System metrics update function


def update_system_metrics() -> None:
    """Update basic host metrics (requires psutil)."""
    try:
        import psutil

        SYSTEM_CPU.set(psutil.cpu_percent())
        SYSTEM_MEM.set(psutil.virtual_memory().percent)
        PROCESS_UPTIME.set(time.monotonic() - _START_TIME)
    except ImportError:
        log.debug("psutil not installed — skipping system metrics update")


# Functions to get metrics output and content type for HTTP response


def get_prometheus_metrics() -> str:
    """Return the latest metrics as plaintext (Prometheus exposition format)."""
    data = generate_latest(REGISTRY)
    return data.decode() if isinstance(data, bytes) else str(data)


def get_metrics_content_type() -> str:
    """Return the appropriate Content-Type for Prometheus metrics."""
    return str(CONTENT_TYPE_LATEST)


__all__ = [
    "CACHE_HITS_TOTAL",
    "CACHE_HIT_RATE",
    "CACHE_MISSES_TOTAL",
    "CONSOLIDATIONS_TOTAL",
    "CONTENT_TYPE_LATEST",
    "CONTRIB_SCORE_HIST",
    "DRIFT_COUNT",
    "EMBEDDING_QUEUE_LENGTH",
    "HEAL_COUNT",
    "LATENCY_BUCKETS",
    "LAT_CONSOLIDATION",
    "LAT_DB_QUERY",
    "LAT_EMBEDDING",
    "LAT_FORGET",
    "LAT_SEARCH",
    "LESSON_PROMOTION_RATE",
    "MEMORY_ACCEPT_RATE",
    "MEM_CREATED_TOTAL",
    "MEM_DELETED_TOTAL",
    "MET_ERRORS_TOTAL",
    "MET_POOL_EXHAUSTED",
    "PROCESS_UPTIME",
    "PURGED_TOTAL",
    "QUALITY_GATE_DROP_REASON",
    "RECALL_AT_K",
    "RETRIEVAL_HIT_AT_K",
    "SYSTEM_CPU",
    "SYSTEM_MEM",
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "generate_latest",
    "get_metrics_content_type",
    "get_prometheus_metrics",
    "measure_time",
    "measure_time_async",
    "metrics_app",
    "prometheus_counter",
    "update_system_metrics",
]
