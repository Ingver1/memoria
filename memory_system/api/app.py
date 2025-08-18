"""
memory_system.api.app
=======================
FastAPI application setup with:
* Lifespan-managed `EnhancedMemoryStore` (no hidden globals).
* OpenTelemetry middleware for distributed tracing.
* Basic `/health/live` and `/health/ready` endpoints for liveness & readiness probes.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, cast

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRouter
from starlette.middleware.base import RequestResponseEndpoint

try:
    import pydantic as _pyd_module
except ImportError:  # pragma: no cover
    from memory_system.utils import pydantic_compat as _pyd_module
ValidationError = _pyd_module.ValidationError

from memory_system import __version__
from memory_system.api.dependencies import get_max_text_length
from memory_system.api.middleware import (
    LanguageDetectionMiddleware,
    MaintenanceModeMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware,
)
from memory_system.api.routes import (
    admin as admin_routes,
    auth as auth_routes,
    health as health_routes,
    memory as memory_routes,
)
from memory_system.api.schemas import MemoryBatchItem, MemoryCreate, MemoryQuery
from memory_system.api.utils import validate_text_length
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.scheduler import scheduler_lifespan
from memory_system.core.store import Memory, get_memory_store
from memory_system.memory_helpers import add, delete, search
from memory_system.settings import UnifiedSettings, configure_logging, get_settings
from memory_system.unified_memory import Memory as UMMemory
from memory_system.utils.security import CryptoContext, EnhancedPIIFilter, start_maintenance

if TYPE_CHECKING:  # Only for type checking, not at runtime
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

fastapi_instrumentor: type[FastAPIInstrumentor] | None
try:  # pragma: no cover - optional dependency
    _fastapi_instr_mod = importlib.import_module("opentelemetry.instrumentation.fastapi")
    fastapi_instrumentor = cast(
        "type[FastAPIInstrumentor]",
        _fastapi_instr_mod.FastAPIInstrumentor,
    )
except ImportError:  # pragma: no cover - optional dependency
    fastapi_instrumentor = None

logger = logging.getLogger(__name__)

LOGO_URL = "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"

TAGS_METADATA = [
    {"name": "Auth", "description": "Authentication operations"},
    {"name": "Memory", "description": "Endpoints for managing memories"},
    {"name": "Health", "description": "Liveness and readiness probes"},
    {"name": "Admin", "description": "Administrative operations"},
]

MEMORY_CREATE_EXAMPLE = {
    "text": "hello world",
    "modality": "text",
    "user_id": "user-123",
}

MEMORY_QUERY_EXAMPLE = {"query": "hello world", "top_k": 5}

MEMORY_BATCH_EXAMPLE = [MEMORY_CREATE_EXAMPLE]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Memory"], prefix="/memory")


@router.post(
    "/add",
    summary="Add memory",
    description="Store a single memory item.",
    response_description="Memory UUID",
    responses={
        200: {
            "description": "Memory created",
            "content": {
                "application/json": {"example": {"id": "123e4567-e89b-12d3-a456-426614174000"}}
            },
        },
        400: {"description": "Text exceeds maximum length"},
        422: {"description": "Validation error"},
    },
)
async def add_memory(
    request: Request, body: MemoryCreate = Body(..., example=MEMORY_CREATE_EXAMPLE)
) -> dict[str, str]:
    """Add a new piece of memory."""
    store = get_memory_store(request)
    modality = body.modality
    metadata = dict(body.metadata or {})
    metadata.setdefault("modality", modality)
    max_len = get_max_text_length(store)
    validate_text_length(body.text, max_len)
    mem = await add(body.text, metadata=metadata, modality=modality, store=store)
    return {"id": mem.memory_id}


@router.delete(
    "/{memory_id}",
    summary="Delete memory",
    description="Remove a memory entry by its ID.",
    response_description="Deletion status",
    responses={
        200: {
            "description": "Memory removed",
            "content": {"application/json": {"example": {"status": "deleted"}}},
        },
        404: {"description": "Memory not found"},
    },
)
async def delete_memory(request: Request, memory_id: str) -> dict[str, str]:
    """Delete a memory entry by ID."""
    store = get_memory_store(request)
    await delete(memory_id, store=store)
    return {"status": "deleted"}


@router.post(
    "/search",
    summary="Search memory",
    description="Semantic search across stored memories.",
    response_description="Search results",
    responses={
        200: {
            "description": "Search results",
            "content": {"application/json": {"example": {"results": []}}},
        },
        400: {"description": "Invalid search query"},
        422: {"description": "Validation error"},
    },
)
async def search_memory(
    request: Request, body: MemoryQuery = Body(..., example=MEMORY_QUERY_EXAMPLE)
) -> Any:
    """Semantic search across stored memories."""
    store = get_memory_store(request)
    return await search(
        body.query,
        k=body.top_k,
        metadata_filter=body.metadata_filter,
        context=body.context,
        modality=body.modality,
        store=store,
    )


@router.post(
    "/batch",
    summary="Add memories batch",
    description="Add multiple memories in a single request.",
    response_description="Memory UUIDs",
    responses={
        200: {
            "description": "Memories created",
            "content": {"application/json": {"example": {"ids": ["id1", "id2"]}}},
        },
        400: {"description": "Invalid memory payload"},
        422: {"description": "Validation error"},
    },
)
async def add_memories_batch(
    request: Request, body: list[MemoryBatchItem] = Body(..., example=MEMORY_BATCH_EXAMPLE)
) -> dict[str, list[str]]:
    """Add multiple memories in a single request."""
    store = get_memory_store(request)
    if isinstance(store, EnhancedMemoryStore):
        mems = await store.add_memories_batch([item.model_dump(exclude_none=True) for item in body])
        return {"ids": [m.id for m in mems]}
    max_len = get_max_text_length(store)
    memories: list[Memory] = []
    for item in body:
        validate_text_length(item.text, max_len)
        meta = dict(item.metadata or {})
        meta.setdefault("modality", item.modality)
        memories.append(
            Memory.new(
                item.text,
                metadata=meta,
                modality=item.modality,
                memory_type=item.memory_type,
                pinned=item.pinned,
                ttl_seconds=item.ttl_seconds,
                last_used=item.last_used,
                success_score=item.success_score,
                decay=item.decay,
            )
        )
    await store.add_many(cast("list[UMMemory]", memories))
    return {"ids": [m.id for m in memories]}


@router.post(
    "/stream",
    summary="Stream memories",
    description="Stream newline-delimited JSON memories to the store.",
    response_description="Count of inserted records",
    responses={
        200: {
            "description": "Number of memories inserted",
            "content": {"application/json": {"example": {"added": 1}}},
        },
        400: {"description": "Invalid JSON or stream limit exceeded"},
        422: {"description": "Validation error"},
    },
)
async def stream_memories(request: Request) -> dict[str, int]:
    """Stream newline-delimited JSON memories to the store."""
    store = get_memory_store(request)
    max_len = get_max_text_length(store)
    max_lines = getattr(
        getattr(getattr(store, "settings", None), "security", None),
        "max_stream_lines",
        10_000,
    )

    line_count = 0

    async def _aiter() -> AsyncIterator[MemoryBatchItem]:
        nonlocal line_count
        async for line in request.stream():
            if not line:
                continue
            line_count += 1
            if line_count > max_lines:
                raise HTTPException(status_code=400, detail="stream limit exceeded")
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="invalid JSON") from exc
            try:
                item = MemoryBatchItem.model_validate(data)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail="validation error") from exc
            validate_text_length(item.text, max_len)
            yield item

    if isinstance(store, EnhancedMemoryStore):
        count = await store.add_memories_streaming(
            item.model_dump(exclude_none=True) async for item in _aiter()
        )
        return {"added": count}

    batch: list[Memory] = []
    count = 0
    async for item in _aiter():
        meta = dict(item.metadata or {})
        batch.append(
            Memory.new(
                item.text,
                metadata=meta,
                modality=item.modality,
                memory_type=item.memory_type,
                pinned=item.pinned,
                ttl_seconds=item.ttl_seconds,
                last_used=item.last_used,
                success_score=item.success_score,
                decay=item.decay,
            )
        )
        if len(batch) >= 100:
            await store.add_many(cast("list[UMMemory]", batch))
            count += len(batch)
            batch.clear()
    if batch:
        await store.add_many(cast("list[UMMemory]", batch))
        count += len(batch)
    return {"added": count}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: UnifiedSettings | None = None) -> FastAPI:  # pragma: no cover
    settings = settings or get_settings()

    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """
        Application lifespan managing core resources.

        Creates the embedding service and memory store (with FAISS index)
        up front and exposes them via ``app.state``.  All components are
        gracefully closed on shutdown.
        """
        try:
            from memory_system.core.embedding import EmbeddingService as _EmbeddingService
        except ModuleNotFoundError:  # optional dependency
            _EmbeddingService = cast("type[Any]", None)
        embedding_service_cls = _EmbeddingService

        if embedding_service_cls is not None:
            try:
                embed_svc: Any | None = embedding_service_cls(settings.model.model_name, settings)
            except ModuleNotFoundError:  # sentence-transformers missing
                embed_svc = None
        else:  # pragma: no cover - EmbeddingService import failed
            embed_svc = None
        pii_filter = EnhancedPIIFilter()
        crypto_ctx = CryptoContext.from_env()
        maintenance_task = await start_maintenance(crypto_ctx)
        try:
            async with EnhancedMemoryStore(settings) as store:
                app.state.embedding_service = embed_svc
                app.state.pii_filter = pii_filter
                app.state.store = store
                app.state.memory_store = store
                app.state.vector_store = getattr(store, "vector_store", None)
                app.state.crypto_ctx = crypto_ctx
                app.state.crypto_maintenance = maintenance_task
                async with scheduler_lifespan(store, settings) as scheduler:
                    app.state.scheduler = scheduler
                    logger.info("EnhancedMemoryStore initialised")
                    try:
                        yield
                    finally:
                        logger.info("EnhancedMemoryStore closed")
                        if embed_svc is not None:
                            with suppress(Exception):
                                await embed_svc.close()
                        vector_store = getattr(store, "vector_store", None)
                        close = getattr(vector_store, "close", None)
                        if close:
                            await close()
        finally:
            maintenance_task.cancel()
            with suppress(Exception):
                await maintenance_task

    app = FastAPI(
        title="Unified Memory System",
        version=__version__,
        description="REST API for storing and querying memories.",
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Maintenance mode middleware
    maintenance = MaintenanceModeMiddleware(app)

    @app.middleware("http")
    async def _maintenance_mw(request: Request, call_next: RequestResponseEndpoint) -> Response:
        return await maintenance.dispatch(request, call_next)

    app.state.maintenance = maintenance

    # Security headers and CORS restrictions
    allowed_origins = [o for o in settings.api.cors_origins if o != "*"]

    app.add_middleware(
        SecurityHeadersMiddleware,
        csp=" ".join(["default-src 'self'", *allowed_origins]),
    )

    # Rate limiting middleware
    app.add_middleware(
        RateLimitingMiddleware,
        max_requests=settings.security.rate_limit_per_minute,
    )

    # Text normalization, language detection and optional translation
    app.add_middleware(
        LanguageDetectionMiddleware,
        translate=settings.api.enable_translation,
        threshold=settings.api.translation_confidence_threshold,
        cache_size=settings.api.translation_cache_size,
    )

    # CORS (restricted origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenTelemetry -------------------------------------------------------
    if fastapi_instrumentor is not None and settings.security.telemetry_level == "aggregate":
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:  # pragma: no cover - optional feature
                from opentelemetry import trace
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import SERVICE_NAME, Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                resource = Resource(attributes={SERVICE_NAME: "memory_system"})
                provider = TracerProvider(resource=resource)
                processor = BatchSpanProcessor(OTLPSpanExporter())
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)
                logger.info("OTLP tracing exporter configured")
            except ImportError:
                logger.warning("opentelemetry-exporter-otlp not installed, tracing disabled")
        fastapi_instrumentor.instrument_app(app)

    # Health probes ---------------------------------------------------------
    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready", include_in_schema=False)
    async def ready(request: Request) -> dict[str, str]:
        store = cast("EnhancedMemoryStore | None", get_memory_store(request))
        if store is None:
            return {"status": "unready"}
        try:
            await store.ping()
            return {"status": "ready"}
        except (OSError, RuntimeError) as exc:  # pylint: disable=broad-except
            logger.error("Readiness check failed: %s", exc)
            return {"status": "unready"}

    # Routers ---------------------------------------------------------------
    # Memory endpoints live under /api/v1/memory
    app.include_router(auth_routes.router, prefix="/api/v1")
    app.include_router(memory_routes.router, prefix="/api/v1/memory")
    app.include_router(health_routes.router, prefix="/api/v1")
    app.include_router(admin_routes.router, prefix="/api/v1")

    @app.get("/")
    async def service_root() -> dict[str, Any]:
        return await health_routes.root()

    @app.get("/health")
    async def health_alias(request: Request) -> Response:
        return await health_routes.health_check(request)

    # Metrics ---------------------------------------------------------------
    if settings.monitoring.enable_metrics and settings.security.telemetry_level == "aggregate":
        try:
            from memory_system.utils.metrics import metrics_app

            app.mount("/metrics", cast("Any", metrics_app))
            logger.info("Prometheus /metrics endpoint enabled")
        except ImportError:  # pragma: no cover - optional feature
            logger.warning("prometheus_client not installed, cannot expose /metrics")

    def _custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=TAGS_METADATA,
        )
        openapi_schema["info"]["x-logo"] = {"url": LOGO_URL}
        app.openapi_schema = openapi_schema
        return openapi_schema

    app.openapi = _custom_openapi

    return app


# ---------------------------------------------------------------------------
# Entry-point for `uvicorn memory_system.api.app:create_app`
# (`uvicorn memory_system.api.app:app` remains for backward compatibility)
# ---------------------------------------------------------------------------
settings = get_settings()
app = create_app(settings)
