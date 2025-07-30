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
from typing import Any, TYPE_CHECKING, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

if TYPE_CHECKING:  # Only for type checking, not at runtime
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as FastAPIInstrumentor

try:  # pragma: no cover - optional dependency
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor as _FastAPIInstrumentor
except Exception:  # pragma: no cover - optional dependency
    _FastAPIInstrumentor = None

fastapi_instrumentor: "FastAPIInstrumentor | None" = cast(
    "FastAPIInstrumentor | None", _FastAPIInstrumentor
)
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

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

    # Maintenance mode middleware
    maintenance = MaintenanceModeMiddleware(app)

    @app.middleware("http")
    async def _maintenance_mw(request: Request, call_next: RequestResponseEndpoint) -> Response:
        return await maintenance.dispatch(request, call_next)

    app.state.maintenance = maintenance

    # CORS (can be tightened in prod)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenTelemetry
    if fastapi_instrumentor is not None and os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        fastapi_instrumentor.instrument_app(app)

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
        app.state.memory_store = await get_store(settings.database.db_path)
        # Dependency bridge ----------------------------------------------------
        app.dependency_overrides[get_memory_store] = lambda req: cast(SQLiteMemoryStore, req.app.state.memory_store)
        logger.info("SQLiteMemoryStore initialised")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        store = cast(SQLiteMemoryStore, app.state.memory_store)
        await store.aclose()
        logger.info("SQLiteMemoryStore closed")

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
            logger.warning("prometheus_client not installed, cannot expose /metrics")

    return app


# ---------------------------------------------------------------------------
# Entry‑point for `uvicorn memory_system.api.app:create_app`
# (`uvicorn memory_system.api.app:app` remains for backward compatibility)
# ---------------------------------------------------------------------------
settings = get_settings()
app = create_app(settings)
