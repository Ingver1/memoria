"""middleware.py — FastAPI middlewares for Unified Memory System.

Includes:
- SessionTracker: in-memory tracker for user activity timestamps.
- RateLimitingMiddleware: token-bucket rate limiting per user/IP.
- MaintenanceModeMiddleware: gate that blocks requests during maintenance mode.
- check_dependencies(): utility to verify optional dependencies at runtime.
"""

from __future__ import annotations

from memory_system import __version__

# Update module docstring with the current version
__doc__ = __doc__.replace("Unified Memory System", f"Unified Memory System (v{__version__})")

import asyncio
import hashlib
import importlib
import logging
import os
import time
from collections import deque
from collections.abc import MutableMapping
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

__all__ = [
    "session_tracker",
    "SessionTracker",
    "RateLimitingMiddleware",
    "MaintenanceModeMiddleware",
    "check_dependencies",
]

log = logging.getLogger(__name__)

# Session tracking helper


class SessionTracker:
    """Thread-safe tracker that records the last activity timestamp per user."""

    _last_seen: MutableMapping[str, float] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def mark(cls, user_id: str) -> None:
        """Register the current UTC timestamp for the given user_id."""
        async with cls._lock:
            cls._last_seen[user_id] = time.time()

    @classmethod
    async def active_count(cls, window_seconds: int = 3600) -> int:
        """Return count of users seen within the last window (seconds)."""
        threshold = time.time() - window_seconds
        async with cls._lock:
            return sum(1 for ts in cls._last_seen.values() if ts >= threshold)

    @classmethod
    def values(cls) -> list[float]:
        """Return a list of all tracked last-seen timestamps."""
        # No lock needed for atomic retrieval of values reference
        return list(cls._last_seen.values())


# Global session tracker instance
session_tracker = SessionTracker()

# Rate limiting middleware


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limiting per user or client IP address."""

    def __init__(
        self,
        app: ASGIApp,
        max_requests: int = 100,
        window_seconds: int = 60,
        bypass_endpoints: Optional[set[str]] = None,
    ) -> None:
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
        # Maps user_id -> deque of request timestamps
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _get_user_id(request: Request) -> str:
        """Derive a stable ID from the Authorization header or client IP."""
        auth = request.headers.get("authorization") or getattr(request.client, "host", "unknown")
        return hashlib.sha256(str(auth).encode()).hexdigest()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for bypassed endpoints
        if request.url.path in self.bypass:
            return await call_next(request)

        user_id = self._get_user_id(request)
        now = time.time()
        async with self._lock:
            bucket = self._hits.setdefault(user_id, deque())
            # Drop timestamps older than the time window
            while bucket and bucket[0] <= now - self.window:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = int(bucket[0] + self.window - now) + 1
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)
        # Track active session
        await session_tracker.mark(user_id)
        return await call_next(request)


# Maintenance mode middleware


class MaintenanceModeMiddleware(BaseHTTPMiddleware):
    """Middleware to block all non-exempt requests when maintenance mode is enabled."""

    def __init__(self, app: ASGIApp, allowed_paths: Optional[set[str]] = None) -> None:
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
        if self._enabled and not any(
            request.url.path.startswith(p) for p in self.allowed_paths
        ):
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
