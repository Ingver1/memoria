"""Stub of prometheus_client for tests without external dependency."""

from contextlib import contextmanager


class _Metric:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        pass

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        pass

    def set(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        pass

    def observe(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        pass

    @contextmanager
    def time(self):  # pragma: no cover - simple stub
        yield


class Counter(_Metric):
    pass


class Gauge(_Metric):
    pass


class Histogram(_Metric):
    pass


CONTENT_TYPE_LATEST = "text/plain"


def generate_latest(*args, **kwargs):  # pragma: no cover - simple stub
    return b""
