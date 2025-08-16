"""
middleware.py — FastAPI middlewares for Unified Memory System.

Includes:
- SessionTracker: in-memory tracker for user activity timestamps.
- RateLimitingMiddleware: token-bucket rate limiting per user/IP.
- MaintenanceModeMiddleware: gate that blocks requests during maintenance mode.
- SecurityHeadersMiddleware: adds common security headers (CSP, HSTS, etc.).
- LanguageDetectionMiddleware: normalizes text and detects language.
- check_dependencies(): utility to verify optional dependencies at runtime.
"""

from __future__ import annotations

from memory_system import __version__

# Update module docstring with the current version
__doc__ = __doc__.replace("Unified Memory System", f"Unified Memory System (v{__version__})")

import asyncio
import base64
import hashlib
import importlib
import json
import logging
import os
import threading
import time
import unicodedata
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

from fastapi import Request  # noqa: TC002
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    from starlette.types import ASGIApp, Message

# ``slowapi`` is optional; provide runtime fallbacks when unavailable
_HAS_SLOWAPI = False
if TYPE_CHECKING:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address
else:  # pragma: no cover - optional dependency
    try:
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        _HAS_SLOWAPI = True
    except Exception:  # noqa: BLE001 - library not installed
        Limiter = cast("Any", None)

        class RateLimitExceeded(Exception):  # noqa: N818
            ...

        def get_remote_address(request: Request) -> str:
            return cast("str", getattr(request.client, "host", "unknown"))

        _HAS_SLOWAPI = False

__all__ = [
    "LanguageDetectionMiddleware",
    "MaintenanceModeMiddleware",
    "RateLimitingMiddleware",
    "SecurityHeadersMiddleware",
    "SessionTracker",
    "check_dependencies",
    "session_tracker",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection utilities
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency
    from pycld3 import NNetLanguageIdentifier

    _CLD3 = NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
except Exception:  # noqa: BLE001
    _CLD3 = None

_FT_MODEL_PATH = os.getenv("FASTTEXT_MODEL_PATH")
if _FT_MODEL_PATH:
    try:  # pragma: no cover - optional dependency
        import fasttext

        _FT_MODEL = fasttext.load_model(_FT_MODEL_PATH)
    except Exception:  # noqa: BLE001
        _FT_MODEL = None
else:
    _FT_MODEL = None


def _detect_language(text: str) -> tuple[str, float]:
    """Return language code and confidence using available detector."""
    if _CLD3 is not None:
        res = _CLD3.FindLanguage(text)
        if res and getattr(res, "is_reliable", False):
            return res.language, float(res.probability)
    if _FT_MODEL is not None:
        labels, probs = _FT_MODEL.predict(text)
        if labels:
            return labels[0].replace("__label__", ""), float(probs[0])
    return "und", 0.0


_TRANSLATOR = None
_SUPPORTED_LANGS = {"en"}

# ---------------------------------------------------------------------------
# Caching utilities for language detection and translation
# ---------------------------------------------------------------------------

_SECRET = os.environ.get("TRANSLATION_CACHE_KEY", "").encode()
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 600.0  # seconds
_CACHE_MAXSIZE = 1024

_LANG_CACHE: OrderedDict[str, tuple[tuple[str, float], float]] = OrderedDict()
_TRANSLATION_CACHE: OrderedDict[str, tuple[str, float]] = OrderedDict()


def _canon(text: str) -> str:
    """Return canonical form of *text* for stable hashing."""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = " ".join(t.split())
    return unicodedata.normalize("NFC", t)


def _cache_key(text: str, src_lang: str = "", dst_lang: str = "") -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    msg = f"{t}{src_lang}{dst_lang}".encode()
    if _SECRET:
        return hashlib.blake2s(msg, key=_SECRET, digest_size=16).hexdigest()
    return hashlib.sha256(msg).hexdigest()


def _cache_get(cache: OrderedDict[str, tuple[Any, float]], key: str) -> Any | None:
    if _CACHE_MAXSIZE <= 0:
        return None
    with _CACHE_LOCK:
        item = cache.get(key)
        if item is None:
            return None
        value, ts = item
        if time.time() - ts > _CACHE_TTL:
            del cache[key]
            return None
        cache.move_to_end(key)
        return value


def _cache_set(cache: OrderedDict[str, tuple[Any, float]], key: str, value: Any) -> None:
    if _CACHE_MAXSIZE <= 0:
        return
    with _CACHE_LOCK:
        cache[key] = (value, time.time())
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAXSIZE:
            cache.popitem(last=False)


def _translate_to_en(text: str, source_lang: str | None = None) -> str:
    """Translate *text* to English using available models."""
    global _TRANSLATOR
    try:  # pragma: no cover - optional dependency
        if _TRANSLATOR is None:
            from transformers import pipeline

            model = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
            _TRANSLATOR = pipeline("translation", model=model)
        kwargs = {"tgt_lang": "eng_Latn"}
        if source_lang:
            kwargs["src_lang"] = source_lang
        result = _TRANSLATOR(text, **kwargs)
        return cast("str", result[0]["translation_text"])
    except Exception:  # noqa: BLE001
        return text


# ---------------------------------------------------------------------------
# Asynchronous translation queue
# ---------------------------------------------------------------------------


@dataclass
class _TranslationJob:
    text: str
    source_lang: str | None
    future: asyncio.Future[str]


_translation_queue: asyncio.Queue[_TranslationJob] | None = None
_translation_worker: asyncio.Task[None] | None = None


async def _translation_worker_loop() -> None:
    assert _translation_queue is not None
    while True:
        job = await _translation_queue.get()
        try:
            result = await asyncio.to_thread(_translate_to_en, job.text, job.source_lang)
            if not job.future.cancelled():
                job.future.set_result(result)
        except Exception as exc:  # pragma: no cover - runtime safety
            if not job.future.cancelled():
                job.future.set_exception(exc)
        finally:
            _translation_queue.task_done()


def _ensure_translation_worker() -> None:
    global _translation_queue, _translation_worker
    if _translation_queue is None:
        _translation_queue = asyncio.Queue()
    if _translation_worker is None or _translation_worker.done():
        _translation_worker = asyncio.create_task(_translation_worker_loop())


async def translate_async(
    text: str, source_lang: str | None = None, timeout: float = 5.0
) -> str | None:
    """Translate text via background worker with timeout."""
    try:
        _ensure_translation_worker()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        assert _translation_queue is not None
        await _translation_queue.put(_TranslationJob(text, source_lang, fut))
        return await asyncio.wait_for(fut, timeout)
    except Exception:  # noqa: BLE001
        log.warning("translation service unavailable", exc_info=True)
        return None


# Session tracking helper


class SessionTracker:
    """Thread-safe tracker that records the last activity timestamp per user."""

    _last_seen: ClassVar[MutableMapping[str, float]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def mark(cls, user_id: str) -> None:
        """Register the current UTC timestamp for the given user_id."""
        async with cls._lock:
            cls._last_seen[user_id] = time.time()

    @classmethod
    async def prune(cls, window_seconds: int) -> None:
        """Remove entries older than ``window_seconds`` from the tracker."""
        threshold = time.time() - window_seconds
        async with cls._lock:
            stale = [uid for uid, ts in cls._last_seen.items() if ts < threshold]
            for uid in stale:
                del cls._last_seen[uid]

    @classmethod
    async def active_count(cls, window_seconds: int = 3600) -> int:
        """Return count of users seen within the last window (seconds)."""
        await cls.prune(window_seconds)
        async with cls._lock:
            return len(cls._last_seen)

    @classmethod
    def values(cls) -> list[float]:
        """Return a list of all tracked last-seen timestamps."""
        # No lock needed for atomic retrieval of values reference
        return list(cls._last_seen.values())


# Global session tracker instance
session_tracker = SessionTracker()

# ---------------------------------------------------------------------------
# Rate limit pseudonymization utilities
# ---------------------------------------------------------------------------

_MIN_HMAC_KEY_LEN = 32
_RL_KEY_B64 = os.getenv("RATE_LIMIT_HMAC_KEY")
RATE_LIMIT_HMAC_KEY = (
    base64.urlsafe_b64decode(_RL_KEY_B64) if _RL_KEY_B64 else os.urandom(_MIN_HMAC_KEY_LEN)
)
if len(RATE_LIMIT_HMAC_KEY) < _MIN_HMAC_KEY_LEN:
    msg = "RATE_LIMIT_HMAC_KEY must be at least 32 bytes"
    raise ValueError(msg)


def pseudonymize_identifier(token: str | None, client_ip: str | None) -> str:
    """
    Derive a deterministic pseudonym for rate limiting (PBKDF2-HMAC-SHA256).

    This is *not* intended for password storage but uses a more computationally
    expensive key-derivation function to avoid fast hash vulnerabilities.
    """
    ident = token or (client_ip or "unknown")
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        ident.encode("utf-8"),
        RATE_LIMIT_HMAC_KEY,
        100_000,
    )
    return dk.hex()


# Rate limiting middleware


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting per user or client IP address.

    Uses :mod:`slowapi` when available and falls back to an in-memory token
    bucket implementation otherwise.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 100,
        window_seconds: int = 60,
        bypass_endpoints: set[str] | None = None,
    ) -> None:
        """Initialize middleware with token bucket parameters."""
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.bypass = bypass_endpoints or {
            "/api/v1/health",
            "/api/v1/version",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        }

        self._use_slowapi = _HAS_SLOWAPI and Limiter is not None
        if self._use_slowapi:

            def key_func(req: Request) -> str:
                token = req.headers.get("X-API-Token")
                addr = get_remote_address(req)
                return token or str(addr)

            self._limiter = Limiter(key_func=key_func)
            self._limit = f"{max_requests}/{window_seconds} second"
        else:
            # Maps user_id -> deque of request timestamps
            self._hits: dict[str, deque[float]] = {}
            self._lock = asyncio.Lock()

    @staticmethod
    def _get_user_id(request: Request) -> str:
        """Derive a stable ID from ``X-API-Token`` or the client IP address."""
        token = request.headers.get("X-API-Token")
        client_ip = getattr(request.client, "host", None)
        return pseudonymize_identifier(token, client_ip)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and enforce the rate limit."""
        if request.url.path in self.bypass:
            return await call_next(request)

        if self._use_slowapi:

            async def _call(req: Request) -> Response:
                return await call_next(req)

            handler: Callable[[Request], Awaitable[Response]] = self._limiter.limit(self._limit)(
                _call
            )
            try:
                resp: Response = await handler(request)
            except RateLimitExceeded as exc:  # pragma: no cover - handled by tests
                retry_after = int(getattr(exc, "retry_after", self.window))
                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                }
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                    headers=headers,
                )
            await session_tracker.mark(self._get_user_id(request))
            resp.headers.setdefault("X-RateLimit-Limit", str(self.max_requests))
            return resp

        user_id = self._get_user_id(request)
        now = time.time()
        async with self._lock:
            bucket = self._hits.setdefault(user_id, deque())
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = int(bucket[0] + self.window - now) + 1
                headers = {
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                }
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                    headers=headers,
                )
            bucket.append(now)
            remaining = self.max_requests - len(bucket)

        await session_tracker.mark(user_id)
        resp = await call_next(request)
        resp.headers.setdefault("X-RateLimit-Limit", str(self.max_requests))
        resp.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        return resp


# Security headers middleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach standard security headers to all responses."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        csp: str | None = None,
        hsts: str | None = None,
        referrer: str | None = None,
        frame: str | None = None,
    ) -> None:
        """Initialize middleware with optional custom header values."""
        super().__init__(app)
        self.csp = csp or "default-src 'self'"
        self.hsts = hsts or "max-age=63072000; includeSubDomains; preload"
        self.referrer = referrer or "no-referrer"
        self.frame = frame or "DENY"
        self.xcto = "nosniff"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Set security headers on the outgoing response."""
        resp = await call_next(request)
        resp.headers.setdefault("Content-Security-Policy", self.csp)
        resp.headers.setdefault("Strict-Transport-Security", self.hsts)
        resp.headers.setdefault("X-Content-Type-Options", self.xcto)
        resp.headers.setdefault("Referrer-Policy", self.referrer)
        resp.headers.setdefault("X-Frame-Options", self.frame)
        return resp


# Text normalization and language detection middleware


class LanguageDetectionMiddleware(BaseHTTPMiddleware):
    """Normalize incoming text, detect language and optionally translate."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        translate: bool = False,
        threshold: float = 0.5,
        cache_size: int = 1024,
    ) -> None:
        super().__init__(app)
        self._translate = translate
        self._threshold = threshold
        global _CACHE_MAXSIZE
        _CACHE_MAXSIZE = cache_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.headers.get("content-type", "").startswith("application/json"):
            body = await request.body()
            if body:
                try:
                    data = json.loads(body)
                except Exception:  # noqa: BLE001
                    data = None
                if isinstance(data, dict):
                    field = None
                    if isinstance(data.get("text"), str):
                        field = "text"
                    elif isinstance(data.get("query"), str):
                        field = "query"
                    if field:
                        norm = _canon(data[field])
                        key = _cache_key(norm)
                        cached = _cache_get(_LANG_CACHE, key)
                        if cached is None:
                            lang, conf = _detect_language(norm)
                            _cache_set(_LANG_CACHE, key, (lang, conf))
                        else:
                            lang, conf = cached
                        data[field] = norm
                        lang_meta = lang if conf >= self._threshold else "und"
                        data["lang"] = lang_meta
                        data["lang_confidence"] = conf
                        translated: str | None = None
                        if self._should_translate(request) and (
                            conf < self._threshold or lang not in _SUPPORTED_LANGS
                        ):
                            t_key = _cache_key(norm, lang, "en")
                            translated = _cache_get(_TRANSLATION_CACHE, t_key)
                            if translated is None:
                                translated = await translate_async(norm, lang)
                                if translated:
                                    _cache_set(_TRANSLATION_CACHE, t_key, translated)
                            if translated:
                                data[field] = translated
                        meta = data.get("metadata") or {}
                        meta["canonical_claim"] = norm
                        meta["canonical_claim_en"] = translated or norm
                        if translated and field == "text":
                            meta["summary"] = norm
                            meta["summary_en"] = translated
                        data["metadata"] = meta
                        new_body = json.dumps(data).encode("utf-8")

                        async def receive() -> Message:
                            return cast(
                                "Message",
                                {
                                    "type": "http.request",
                                    "body": new_body,
                                    "more_body": False,
                                },
                            )

                        request._receive = receive

        return await call_next(request)

    def _should_translate(self, request: Request) -> bool:
        """Return True if translation is enabled for the current request."""
        if not self._translate:
            return False
        settings = getattr(getattr(request.app, "state", None), "settings", None)
        if settings is not None:
            api_cfg = getattr(settings, "api", None)
            if api_cfg is not None and getattr(api_cfg, "enable_translation", True) is False:
                return False
        return True


# Maintenance mode middleware


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """Middleware to block all non-exempt requests when maintenance mode is enabled."""

    def __init__(self, app: ASGIApp, allowed_paths: set[str] | None = None) -> None:
        """Initialize maintenance gate with allowed paths."""
        super().__init__(app)
        if hasattr(app, "state"):
            app.state.maintenance = self
        # Paths that are always allowed even during maintenance (e.g. admin toggle)
        self.allowed_paths: set[str] = allowed_paths or {
            "/api/v1/admin/maintenance-mode",
            "/health",
            "/healthz",
            "/readyz",
        }
        self._enabled: bool = os.getenv("UMS_MAINTENANCE", "0") == "1"

    def enable(self) -> None:
        """Enable maintenance mode (start rejecting non-exempt requests)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable maintenance mode (resume normal operation)."""
        self._enabled = False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Reject requests when maintenance mode is enabled."""
        if self._enabled and not any(request.url.path.startswith(p) for p in self.allowed_paths):
            # Return 503 Service Unavailable for blocked requests
            return JSONResponse(
                status_code=503,
                content={"detail": "Service is under maintenance, please try later."},
            )
        return await call_next(request)


# Dependency checker for health endpoints


async def check_dependencies() -> dict[str, bool]:
    """Check optional dependencies (like psutil, etc.) and return their availability."""
    results: dict[str, bool] = {}
    # Example: check if psutil is installed
    try:
        importlib.import_module("psutil")
        results["psutil"] = True
    except ImportError:
        results["psutil"] = False
    # Additional dependency checks can be added here
    return results
