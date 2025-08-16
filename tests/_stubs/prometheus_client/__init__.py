class _Metric:
    def __init__(self):
        self._value = 0.0

    def labels(self, *a, **k):
        return self

    def inc(self, amount=1.0):
        if amount < 0:
            raise ValueError("Counters cannot be decremented")
        self._value += amount

    def observe(self, v):
        self._value += v

    def set(self, v):
        self._value = v

    @property
    def value(self):  # pragma: no cover - simple accessor
        return self._value

    class _Timer:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def time(self):
        return self._Timer()


def Counter(*a, **k):
    return _Metric()


def Gauge(*a, **k):
    return _Metric()


def Histogram(*a, **k):
    return _Metric()


class CollectorRegistry:
    def __init__(self):
        self._metrics = {}


# ``prometheus_client`` exposes a module level ``REGISTRY`` which our code
# expects to import.  The lack of this attribute caused ``ImportError`` and
# forced the application to fall back to no-op metrics.  Provide a simple
# instance here so imports succeed during tests.
REGISTRY = CollectorRegistry()


CONTENT_TYPE_LATEST = "text/plain"


def generate_latest(*a, **k):
    return b""


def make_asgi_app(*a, **k):
    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    return _app
