"""memory_system.api.app
=======================
FastAPI application setup with:
* Lifespan‑managed `EnhancedMemoryStore` (no hidden globals).
* OpenTelemetry middleware for distributed tracing.
* Basic `/health/live` and `/health/ready` endpoints for liveness & readiness probes.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncIterator, cast

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest

if TYPE_CHECKING:  # Only for type checking, not at runtime
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

fastapi_instrumentor: "type[FastAPIInstrumentor] | None"
try:  # pragma: no cover - optional dependency
    _fastapi_instr_mod = importlib.import_module("opentelemetry.instrumentation.fastapi")
    fastapi_instrumentor = cast(
        "type[FastAPIInstrumentor]",
        _fastapi_instr_mod.FastAPIInstrumentor,
    )
except Exception:  # pragma: no cover - optional dependency
    fastapi_instrumentor = None

# Allow tests to set ``Request.app`` manually for middleware checks
if not hasattr(StarletteRequest, "_ums_app_setter"):

    def _set_app(self: StarletteRequest, value: FastAPI) -> None:
        self.scope["app"] = value

    StarletteRequest.app = property(lambda self: self.scope.get("app"), _set_app)
    StarletteRequest._ums_app_setter = True

from memory_system import __version__
from memory_system.api.middleware import MaintenanceModeMiddleware, RateLimitingMiddleware
from memory_system.api.routes import admin as admin_routes
from memory_system.api.routes import health as health_routes
from memory_system.api.routes import memory as memory_routes
from memory_system.config.settings import (
    UnifiedSettings,
    configure_logging,
    get_settings,
)
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.store import Memory, get_memory_store
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
    modality = body.get("modality", "text")
    metadata = body.get("metadata", {})
    metadata.setdefault("modality", modality)
    mem = await add(body["text"], metadata=metadata, modality=modality, store=store)
    return {"id": mem.memory_id}


@router.delete("/{memory_id}", summary="Delete memory", response_description="Deletion status")
async def delete_memory(request: Request, memory_id: str) -> dict[str, str]:
    """Delete a memory entry by ID."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    await delete(memory_id, store=store)
    return {"status": "deleted"}


@router.get("/search", summary="Search memory", response_description="Search results")
async def search_memory(
    request: Request,
    q: str,
    limit: int = 5,
    metadata: str | None = None,
    modality: str = "text",
) -> Any:
    """Semantic search across stored memories."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    metadata_filter = json.loads(metadata) if metadata else None
    return await search(q, k=limit, metadata_filter=metadata_filter, modality=modality, store=store)


@router.post("/batch", summary="Add memories batch", response_description="Memory UUIDs")
async def add_memories_batch(request: Request, body: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Add multiple memories in a single request."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))
    if isinstance(store, EnhancedMemoryStore):
        mems = await store.add_memories_batch(body)
        return {"ids": [m.id for m in mems]}
    memories = [Memory.new(item["text"], metadata=item.get("metadata", {})) for item in body]
    await store.add_many(memories)  # type: ignore[attr-defined]
    return {"ids": [m.id for m in memories]}


@router.post("/stream", summary="Stream memories", response_description="Count of inserted records")
async def stream_memories(request: Request) -> dict[str, int]:
    """Stream newline-delimited JSON memories to the store."""
    store = cast(MemoryStoreProtocol, get_memory_store(request))

    async def _aiter() -> AsyncIterator[dict[str, Any]]:
        async for line in request.stream():
            if not line:
                continue
            yield json.loads(line)

    if isinstance(store, EnhancedMemoryStore):
        count = await store.add_memories_streaming(_aiter())
        return {"added": count}

    batch: list[Memory] = []
    count = 0
    async for item in _aiter():
        batch.append(Memory.new(item["text"], metadata=item.get("metadata", {})))
        if len(batch) >= 100:
            await store.add_many(batch)  # type: ignore[attr-defined]
            count += len(batch)
            batch.clear()
    if batch:
        await store.add_many(batch)  # type: ignore[attr-defined]
        count += len(batch)
    return {"added": count}


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

    # Rate limiting middleware
    app.add_middleware(
        RateLimitingMiddleware,
        max_requests=settings.security.rate_limit_per_minute,
    )

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
            store: EnhancedMemoryStore = get_memory_store(request)
            await store.ping()
            return {"status": "ready"}
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Readiness check failed: %s", exc)
            return {"status": "unready"}

    # Lifespan --------------------------------------------------------------
    @app.on_event("startup")
    async def _startup() -> None:
        store = EnhancedMemoryStore(settings)
        app.state.store = store
        app.state.memory_store = store
        # Dependency bridge ----------------------------------------------------
        app.dependency_overrides[get_memory_store] = lambda req: cast(EnhancedMemoryStore, req.app.state.memory_store)
        logger.info("EnhancedMemoryStore initialised")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        store = cast(EnhancedMemoryStore, app.state.store)
        await store.close()
        logger.info("EnhancedMemoryStore closed")

    # Routers ---------------------------------------------------------------
    # Memory endpoints live under /api/v1/memory
    app.include_router(memory_routes.router, prefix="/api/v1/memory")
    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(admin_routes.router, prefix="/api/v1")

    @app.get("/")
    async def service_root() -> dict[str, Any]:
        return cast(dict[str, Any], await health_routes.root())

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
