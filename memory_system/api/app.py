"""memory_system.api.app
=======================
FastAPI application setup with:
* Lifespan‑managed `SQLiteMemoryStore` (no hidden globals).
* OpenTelemetry middleware for distributed tracing.
* Basic `/health/live` and `/health/ready` endpoints for liveness & readiness probes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from memory_system import __version__
from memory_system.api.middleware import MaintenanceModeMiddleware
from memory_system.api.routes import admin as admin_routes
from memory_system.api.routes import health as health_routes
from memory_system.api.routes import memory as memory_routes
from memory_system.config.settings import (
    UnifiedSettings,
    configure_logging,
    get_settings,
)
from memory_system.core.store import SQLiteMemoryStore, get_memory_store, get_store
from memory_system.memory_helpers import MemoryStoreProtocol, add, delete, search
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Memory"], prefix="/memory")


@router.post("/add", summary="Add memory", response_description="Memory UUID")
async def add_memory(request: Request, body: dict[str, Any]) -> dict[str, str]:
    """Add a new piece of memory."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    mem = await add(body["text"], metadata=body.get("metadata", {}), store=store)
    return {"id": mem.memory_id}

@router.delete("/{memory_id}", summary="Delete memory", response_description="Deletion status")
async def delete_memory(request: Request, memory_id: str) -> dict[str, str]:
    """Delete a memory entry by ID."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    await delete(memory_id, store=store)
    return {"status": "deleted"}


@router.get("/search", summary="Search memory", response_description="Search results")
async def search_memory(request: Request, q: str, limit: int = 5) -> Any:
    """Semantic search across stored memories."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    return await search(q, k=limit, store=store)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: UnifiedSettings | None = None) -> FastAPI:  # pragma: no cover
    settings = settings or get_settings()

    configure_logging(settings)

    app = FastAPI(title="Unified Memory System", version=__version__)
    
    # Maintenance mode middleware is attached via middleware stack
    app.add_middleware(MaintenanceModeMiddleware)
    
    # CORS (can be tightened in prod)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenTelemetry
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        FastAPIInstrumentor.instrument_app(app)

    # Health probes ---------------------------------------------------------
    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> dict[str, str]:
        try:
            store: SQLiteMemoryStore = get_memory_store(request)
            await store.ping()
            return {"status": "ready"}
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Readiness check failed: %s", exc)
            return {"status": "unready"}

    # Lifespan --------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        app.state.store = await get_store(settings.database.db_path)
        logger.info("SQLiteMemoryStore initialised")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        store = cast(SQLiteMemoryStore, app.state.store)
        await store.aclose()
        logger.info("SQLiteMemoryStore closed")


    # Dependency bridge -----------------------------------------------------
    app.dependency_overrides[get_memory_store] = (
        lambda req: cast(SQLiteMemoryStore, req.app.state.store)
    )

    # Routers ---------------------------------------------------------------
    # Memory endpoints live under /api/v1/memory
    app.include_router(memory_routes.router, prefix="/api/v1/memory")
    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(admin_routes.router, prefix="/api/v1")

    @app.get("/")
    async def service_root() -> dict[str, Any]:
        return await health_routes.root()

    @app.get("/health")
    async def health_alias() -> Response:
        return await health_routes.health_check()
        
    # Metrics ---------------------------------------------------------------
    if settings.monitoring.enable_metrics:
        try:
            from prometheus_client import make_asgi_app

            app.mount("/metrics", make_asgi_app())
            logger.info("Prometheus /metrics endpoint enabled")
        except ImportError:  # pragma: no cover - optional feature
            logger.warning(
                "prometheus_client not installed, cannot expose /metrics"
            )

    return app


# ---------------------------------------------------------------------------
# Entry‑point for `uvicorn memory_system.api.app:app`
# ---------------------------------------------------------------------------
settings = get_settings()
app = create_app(settings)
