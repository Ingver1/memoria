"""Unified Memory System — Prometheus metrics utilities."""

from __future__ import annotations

from memory_system import __version__

# Update module docstring with current version
__doc__ = __doc__.replace("utilities", f"utilities (v{__version__})")

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

log = logging.getLogger(__name__)


# Helper for creating counters (not used for fixed metrics below, but available for dynamic creation)
def prometheus_counter(name: str, description: str, labels: list[str] | None = None) -> Counter:
    """Create a Prometheus Counter with optional label names."""
    if labels:
        return Counter(name, description, labels)
    return Counter(name, description)


# Base metrics collectors
MET_ERRORS_TOTAL = Counter("ums_errors_total", "Total errors", ["type", "component"])
LAT_DB_QUERY = Histogram("ums_db_query_latency_seconds", "DB query latency")
LAT_SEARCH = Histogram("ums_search_latency_seconds", "Vector search latency")
LAT_EMBEDDING = Histogram("ums_embedding_latency_seconds", "Embedding generation latency")
MET_POOL_EXHAUSTED = Counter("ums_pool_exhausted_total", "Connection pool exhausted events")

# Cache and queue metrics
CACHE_HITS_TOTAL = Counter("ums_cache_hits_total", "Cache hits")
CACHE_MISSES_TOTAL = Counter("ums_cache_misses_total", "Cache misses")
CACHE_HIT_RATE = Gauge("ums_cache_hit_rate", "Cache hit rate")
EMBEDDING_QUEUE_LENGTH = Gauge("ums_embedding_queue_length", "Embedding queue length")

# System metrics gauges
SYSTEM_CPU = Gauge("system_cpu_percent", "CPU usage percentage")
SYSTEM_MEM = Gauge("system_mem_percent", "Memory usage percentage")
PROCESS_UPTIME = Gauge("process_uptime_seconds", "Process uptime in seconds")

_START_TIME = time.monotonic()
PROCESS_UPTIME.set(0.0)

# Timing decorators for measuring execution time of functions


P = ParamSpec("P")
R = TypeVar("R")


def _wrap_sync(metric: Histogram) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator factory for synchronous function timing."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with metric.time():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def _wrap_async(
    metric: Histogram,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """Decorator factory for async function timing."""

    def decorator(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with metric.time():
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
    data = generate_latest()
    return data.decode() if isinstance(data, bytes) else str(data)


def get_metrics_content_type() -> str:
    """Return the appropriate Content-Type for Prometheus metrics."""
    return str(CONTENT_TYPE_LATEST)
