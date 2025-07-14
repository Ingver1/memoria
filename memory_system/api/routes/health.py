"""Health-check and monitoring endpoints."""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from memory_system import __version__
from memory_system.api.dependencies import get_memory_store
from memory_system.api.middleware import check_dependencies, session_tracker
from memory_system.api.schemas import HealthResponse, StatsResponse
from memory_system.config.settings import UnifiedSettings, get_settings
from memory_system.core.store import EnhancedMemoryStore
from memory_system.utils.metrics import get_metrics_content_type, get_prometheus_metrics
from starlette.responses import JSONResponse, Response

log = logging.getLogger(__name__)
router = APIRouter(tags=["Health & Monitoring"])

# Dependency helpers for route functions


async def _store() -> EnhancedMemoryStore:
    """Dependency to get the global EnhancedMemoryStore (async)."""
    return get_memory_store()
    

def _settings() -> UnifiedSettings:
    """Dependency to get current UnifiedSettings."""
    return get_settings()


# Root endpoint (basic service info)


@router.get("/", summary="Service info")
async def root() -> Dict[str, Any]:
    """Root health endpoint providing service information."""
    return {
        "service": "Unified Memory System",
        "version": __version__,
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "api_version": "v1",
    }


# Full health check and liveness/readiness probes


@router.get("/health", summary="Full health check")
async def health_check() -> Response:
    """Return basic health information including component checks."""
    store = await _store()
    component = await store.get_health()
    payload = HealthResponse(
        status="healthy" if component.healthy else "unhealthy",
        timestamp=datetime.now(UTC).isoformat(),
        uptime_seconds=component.uptime,
        version=__version__,
        checks=component.checks,
        memory_store_health={"uptime": component.uptime},
        api_enabled=True,
    )
    status_code = 200 if component.healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    data = payload.model_dump()
    data["healthy"] = component.healthy
    return JSONResponse(
        data,
        status_code=status_code,
        headers={"content-type": "application/json"},
    )

@router.post("/health")
async def health_method_not_allowed() -> JSONResponse:
    """Explicit 405 response for unsupported POST method."""
    return JSONResponse(
        status_code=405, headers={"content-type": "application/json"}
    )


@router.get("/health/live", summary="Liveness probe")
async def liveness_probe() -> Dict[str, str]:
    """Simple liveness probe endpoint (always returns alive if reachable)."""
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/ready", summary="Readiness probe")
async def readiness_probe(
    memory_store: Optional[EnhancedMemoryStore] = None,
) -> Dict[str, Any]:
    """Readiness probe to check if the memory store is ready for requests."""
    store = memory_store if memory_store is not None else await _store()
    component = await store.get_health()
    if component.healthy:
        return {"status": "ready", "timestamp": datetime.now(UTC).isoformat()}
    raise HTTPException(status_code=503, detail=f"Service not ready: {component.message}")


@router.get("/stats", summary="System statistics")
async def get_stats(
    memory_store: Optional[EnhancedMemoryStore] = None,
    settings: Optional[UnifiedSettings] = None,
) -> Dict[str, Any]:
    """Retrieve current system and memory store statistics."""
    store = memory_store if memory_store is not None else await _store()
    config = settings if settings is not None else _settings()
    if asyncio.iscoroutine(config):
        config = await config
    stats = await store.get_stats()
    current_time = datetime.now(UTC).timestamp()
    active = sum(1 for ts in session_tracker.values() if ts > current_time - 3600)
    payload = StatsResponse(
        total_memories=stats.get("total_memories", 0),
        active_sessions=active,
        uptime_seconds=stats.get("uptime_seconds", 0),
        memory_store_stats=stats,
        api_stats={
            "cors_enabled": config.api.enable_cors,
            "rate_limiting_enabled": config.monitoring.enable_rate_limiting,
            "metrics_enabled": config.monitoring.enable_metrics,
            "encryption_enabled": config.security.encrypt_at_rest,
            "pii_filtering_enabled": config.security.filter_pii,
            "backup_enabled": config.reliability.backup_enabled,
            "model_name": config.model.model_name,
            "api_version": "v1",
        },
    )
    return payload.model_dump()


@router.get("/metrics", summary="Prometheus metrics")
async def metrics_endpoint(settings: Optional[UnifiedSettings] = None) -> Response:
    """Expose Prometheus metrics if enabled, otherwise 404."""
    settings = settings or _settings()
    if asyncio.iscoroutine(settings):
        settings = await settings
    if not settings.monitoring.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    ctype = get_metrics_content_type()
    return Response(
        content=get_prometheus_metrics(),
        media_type=ctype,
        headers={"content-type": ctype},
    )


@router.get("/version", summary="Version info")
async def get_version() -> Dict[str, Any]:
    """Get version and environment details of the running service."""
    return {
        "version": __version__,
        "api_version": "v1",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }
