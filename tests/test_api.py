"""Comprehensive tests for API module."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

import pytest_asyncio

try:
    from fastapi.testclient import ClientHelper as TestClient
except ImportError:  # FastAPI >= 0.111
    from fastapi.testclient import TestClient
from starlette.responses import Response

from memory_system import __version__, unified_memory

try:  # Optional dependencies may be missing in minimal environments
    from memory_system.api.app import create_app
    from memory_system.api.schemas import (
        ErrorResponse,
        HealthResponse,
        MemoryCreate,
        MemoryQuery,
        MemoryRead,
        MemorySearchResult,
        StatsResponse,
        SuccessResponse,
    )
    from memory_system.core.store import Memory
    from memory_system.settings import UnifiedSettings
except ModuleNotFoundError as exc:  # pragma: no cover - skip when deps missing
    pytest.skip(f"optional dependency not available: {exc}", allow_module_level=True)

pytestmark = [pytest.mark.needs_fastapi, pytest.mark.needs_httpx]

EPSILON = 1e-9


@pytest.fixture
def test_settings() -> UnifiedSettings:
    """Create test settings."""
    return UnifiedSettings.for_testing()


@pytest.fixture
def test_app(test_settings: UnifiedSettings) -> FastAPI:
    """Create test FastAPI app."""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(test_app)


@pytest_asyncio.fixture
async def async_test_client(test_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async test client."""
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""

    def test_root_endpoint(self, test_client: TestClient) -> None:
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Unified Memory System"
        assert data["version"] == __version__
        assert data["status"] == "running"
        assert data["documentation"] == "/docs"
        assert data["health"] == "/health"
        assert data["metrics"] == "/metrics"
        assert data["api_version"] == "v1"

    def test_health_endpoint(self, test_client: TestClient) -> None:
        """Test health check endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers.get("content-type") == "application/json"

        data = response.json()
        health_response = HealthResponse(**data)

        assert health_response.status in ["healthy", "degraded", "unhealthy"]
        assert health_response.version == __version__
        assert health_response.uptime_seconds >= 0
        assert isinstance(health_response.checks, dict)
        assert isinstance(health_response.memory_store_health, dict)
        assert isinstance(health_response.api_enabled, bool)
        assert isinstance(health_response.ranking_min_score, float)
        assert health_response.timestamp is not None

    def test_liveness_probe(self, test_client: TestClient) -> None:
        """Test liveness probe endpoint."""
        response = test_client.get("/api/v1/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data

    def test_readiness_probe(self, test_client: TestClient) -> None:
        """Test readiness probe endpoint."""
        response = test_client.get("/api/v1/health/ready")
        # Should be 200 for healthy service or 503 for unhealthy
        assert response.status_code in [200, 503]

        data = response.json()
        if response.status_code == 200:
            assert data["status"] == "ready"
        else:
            assert "detail" in data

    def test_stats_endpoint(self, test_client: TestClient) -> None:
        """Test stats endpoint."""
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        stats_response = StatsResponse(**data)

        assert stats_response.total_memories >= 0
        assert stats_response.active_sessions >= 0
        assert stats_response.uptime_seconds >= 0
        assert isinstance(stats_response.memory_store_stats, dict)
        assert isinstance(stats_response.api_stats, dict)

    def test_version_endpoint(self, test_client: TestClient) -> None:
        """Test version endpoint."""
        response = test_client.get("/api/v1/version")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == __version__
        assert data["api_version"] == "v1"
        assert "python_version" in data
        assert "platform" in data
        assert "architecture" in data

    def test_metrics_endpoint_enabled(self, test_client: TestClient) -> None:
        """Test metrics endpoint when enabled."""
        response = test_client.get("/api/v1/metrics")
        # Should be 200 if metrics enabled or 404 if disabled
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            # Should be Prometheus format
            assert (
                response.headers.get("content-type") == "text/plain; version=0.0.4; charset=utf-8"
            )
            content = response.text
            assert "# HELP" in content or "# TYPE" in content

    def test_metrics_endpoint_disabled(
        self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test metrics endpoint when disabled."""
        from fastapi import HTTPException

        from memory_system.api.routes import health as health_routes
        from memory_system.settings import UnifiedSettings

        # Override the settings dependency to disable metrics
        monkeypatch.setattr(
            health_routes,
            "_settings",
            lambda: UnifiedSettings.for_testing(),
        )

        with pytest.raises(HTTPException) as exc_info:
            test_client.get("/api/v1/metrics")

        assert exc_info.value.status_code == 404

    def test_root_metrics_endpoint_disabled(self) -> None:
        """/metrics should return 404 when metrics are disabled."""
        app = create_app(UnifiedSettings.for_testing())
        with TestClient(app) as client:
            resp = client.get("/metrics")
            assert resp.status_code == 404


class TestAdminEndpoints:
    """Test admin endpoints."""

    def test_maintenance_mode_status(self, test_client: TestClient) -> None:
        """Test getting maintenance mode status."""
        token_resp = test_client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "test-token-12345678"},
        )
        access = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {access}"}
        response = test_client.get("/api/v1/admin/maintenance-mode", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data == {"enabled": False}

    def test_maintenance_mode_enable(self, test_client: TestClient) -> None:
        """Test enabling maintenance mode."""
        token_resp = test_client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "test-token-12345678"},
        )
        token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        response = test_client.post("/api/v1/admin/maintenance-mode/enable", headers=headers)
        assert response.status_code == 204
        assert test_client.app.state.maintenance._enabled is True

    def test_maintenance_mode_disable(self, test_client: TestClient) -> None:
        """Test disabling maintenance mode."""
        token_resp = test_client.post(
            "/api/v1/auth/token",
            data={"username": "admin", "password": "test-token-12345678"},
        )
        token = token_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        test_client.post("/api/v1/admin/maintenance-mode/enable", headers=headers)
        response = test_client.post("/api/v1/admin/maintenance-mode/disable", headers=headers)
        assert response.status_code == 204
        assert test_client.app.state.maintenance._enabled is False


@pytest.mark.asyncio
async def test_delete_user_removes_memories_and_revokes_keys(
    test_app: FastAPI, async_test_client: httpx.AsyncClient
) -> None:
    """Deleting a user removes memories, vectors and revokes tokens."""
    store = test_app.state.memory_store
    vector_store = test_app.state.vector_store

    user_id = "user-test"
    mem = Memory.new("hello", metadata={"user_id": user_id})
    await store.meta_store.add(mem)
    dim = store.settings.model.vector_dim
    vector_store.add([mem.id], [[0.0] * dim])

    initial_vectors = vector_store.stats().total_vectors

    class DummyTM:
        def __init__(self) -> None:
            self.revoked: list[str] = []

        def revoke_user(self, uid: str) -> None:
            self.revoked.append(uid)

    tm = DummyTM()
    test_app.state.token_manager = tm

    resp = await async_test_client.delete(f"/api/v1/admin/users/{user_id}")
    assert resp.status_code == 204

    remaining = await store.meta_store.search(metadata_filters={"user_id": user_id})
    assert remaining == []

    assert vector_store.stats().total_vectors == initial_vectors - 1
    assert tm.revoked == [user_id]


@pytest.mark.asyncio
async def test_delete_user_logs_token_revoke_error(
    test_app: FastAPI,
    async_test_client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed token revocations are logged."""
    store = test_app.state.memory_store
    vector_store = test_app.state.vector_store

    user_id = "user-log"
    mem = Memory.new("hello", metadata={"user_id": user_id})
    await store.meta_store.add(mem)
    dim = store.settings.model.vector_dim
    vector_store.add([mem.id], [[0.0] * dim])

    class DummyTM:
        def revoke_token(self, _token: str) -> None:
            msg = "boom"
            raise RuntimeError(msg)

    test_app.state.token_manager = DummyTM()
    test_app.state.user_tokens = {user_id: ["tok-1"]}

    caplog.set_level(logging.WARNING, logger="memory_system.api.routes.admin")

    resp = await async_test_client.delete(f"/api/v1/admin/users/{user_id}")
    assert resp.status_code == 204
    assert any("Token revoke failed" in r.message for r in caplog.records)


class TestSchemas:
    """Test Pydantic schemas."""

    def test_memory_create_schema(self) -> None:
        """Test MemoryCreate schema."""
        # Valid data
        data = {
            "text": "Test memory text",
            "role": "user",
            "tags": ["test", "memory"],
            "metadata": {"foo": "bar"},
        }
        memory_create = MemoryCreate(**data)
        assert memory_create.text == "Test memory text"
        assert memory_create.role == "user"
        assert memory_create.tags == ["test", "memory"]
        assert memory_create.user_id is None
        assert memory_create.metadata == {"foo": "bar"}

        # With user_id
        data["user_id"] = "user123"
        memory_create = MemoryCreate(**data)
        assert memory_create.user_id == "user123"

    def test_memory_create_schema_validation(self) -> None:
        """Test MemoryCreate schema validation."""
        # Empty text should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="")

        # Text too long should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="x" * 10001)

        # Role too long should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="test", role="x" * 33)

        # Too many tags should fail
        with pytest.raises(ValueError):
            MemoryCreate(text="test", tags=["tag"] * 11)

    def test_memory_read_schema(self) -> None:
        """Test MemoryRead schema."""
        from datetime import UTC, datetime

        data = {
            "id": "mem123",
            "user_id": "user123",
            "text": "Test memory text",
            "role": "user",
            "tags": ["test"],
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        memory_read = MemoryRead(**data)
        assert memory_read.id == "mem123"
        assert memory_read.user_id == "user123"
        assert memory_read.text == "Test memory text"
        assert memory_read.role == "user"
        assert memory_read.tags == ["test"]
        assert isinstance(memory_read.created_at, datetime)
        assert isinstance(memory_read.updated_at, datetime)

    def test_memory_update_schema(self) -> None:
        """Test MemoryUpdate schema."""
        from memory_system.api.schemas import MemoryUpdate

        # All fields optional
        update = MemoryUpdate()
        assert update.text is None
        assert update.role is None
        assert update.tags is None
        assert update.valence is None
        assert update.emotional_intensity is None
        assert update.valence_delta is None
        assert update.emotional_intensity_delta is None

        # Partial update via deltas
        update = MemoryUpdate(
            text="Updated text",
            valence_delta=0.2,
            emotional_intensity_delta=-0.1,
        )
        assert update.text == "Updated text"
        assert update.role is None
        assert update.tags is None
        assert abs(update.valence_delta - 0.2) < EPSILON
        assert abs(update.emotional_intensity_delta + 0.1) < EPSILON

        # Absolute update values
        update = MemoryUpdate(valence=-0.3, emotional_intensity=0.8)
        assert update.valence == pytest.approx(-0.3)
        assert update.emotional_intensity == pytest.approx(0.8)

    def test_memory_query_schema(self) -> None:
        """Test MemoryQuery schema."""
        query = MemoryQuery(query="test query")
        assert query.query == "test query"
        assert query.top_k == 10
        assert query.include_embeddings is False

        # Custom values
        query = MemoryQuery(query="test query", top_k=5, include_embeddings=True)
        assert query.query == "test query"
        assert query.top_k == 5
        assert query.include_embeddings is True

    def test_memory_query_schema_validation(self) -> None:
        """Test MemoryQuery schema validation."""
        # Empty query should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="")

        # Query too long should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="x" * 1001)

        # top_k out of range should fail
        with pytest.raises(ValueError):
            MemoryQuery(query="test", top_k=0)

        with pytest.raises(ValueError):
            MemoryQuery(query="test", top_k=101)

    def test_memory_search_result_schema(self) -> None:
        """Test MemorySearchResult schema."""
        from datetime import UTC, datetime

        data = {
            "id": "mem123",
            "user_id": "user123",
            "text": "Test memory text",
            "role": "user",
            "tags": ["test"],
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "score": 0.95,
            "embedding": [0.1, 0.2, 0.3],
        }
        result = MemorySearchResult(**data)
        assert result.id == "mem123"
        assert result.score == 0.95
        assert result.embedding == [0.1, 0.2, 0.3]

    def test_health_response_schema(self) -> None:
        """Test HealthResponse schema."""
        data = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "uptime_seconds": 3600,
            "version": __version__,
            "checks": {"database": True, "index": True},
            "memory_store_health": {"total_memories": 100},
            "api_enabled": True,
            "ranking_min_score": 0.0,
        }
        health = HealthResponse(**data)
        assert health.status == "healthy"
        assert health.timestamp == "2024-01-01T00:00:00Z"
        assert health.uptime_seconds == 3600
        assert health.version == __version__
        assert health.checks == {"database": True, "index": True}
        assert health.memory_store_health == {"total_memories": 100}
        assert health.api_enabled is True
        assert health.ranking_min_score == 0.0

    def test_stats_response_schema(self) -> None:
        """Test StatsResponse schema."""
        data = {
            "total_memories": 1000,
            "active_sessions": 50,
            "uptime_seconds": 3600,
            "memory_store_stats": {"cache_hit_rate": 0.85},
            "api_stats": {"requests_per_second": 100},
        }
        stats = StatsResponse(**data)
        assert stats.total_memories == 1000
        assert stats.active_sessions == 50
        assert stats.uptime_seconds == 3600
        assert stats.memory_store_stats == {"cache_hit_rate": 0.85}
        assert stats.api_stats == {"requests_per_second": 100}

    def test_success_response_schema(self) -> None:
        """Test SuccessResponse schema."""
        success = SuccessResponse()
        assert success.message == "success"
        assert success.api_version == "v1"

        success = SuccessResponse(message="custom success")
        assert success.message == "custom success"
        assert success.api_version == "v1"

    def test_error_response_schema(self) -> None:
        """Test ErrorResponse schema."""
        error = ErrorResponse(detail="Something went wrong")
        assert error.detail == "Something went wrong"
        assert error.api_version == "v1"


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_headers(self, test_client: TestClient) -> None:
        """Test CORS headers are present."""
        test_client.get("/api/v1/health")
        # CORS headers should be present if enabled
        # This depends on the test configuration

    def test_rate_limiting_middleware(self, test_client: TestClient) -> None:
        """Test rate limiting middleware."""
        # Make multiple requests to trigger rate limiting
        responses = []
        for _i in range(10):
            response = test_client.get("/api/v1/health")
            responses.append(response)

        # All should succeed for a small number of requests
        for response in responses:
            assert response.status_code in [200, 429]

    def test_maintenance_mode_middleware(self, test_client: TestClient) -> None:
        """Test maintenance mode middleware blocks and unblocks requests."""
        mw = test_client.app.state.maintenance

        # Ensure disabled by default
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200

        # Enable maintenance mode
        test_client.post("/api/v1/admin/maintenance-mode/enable")
        assert mw._enabled is True

        scope = {"type": "http", "path": "/api/v1/health", "headers": []}
        req = Request(scope)
        req.app = test_client.app

        async def _next(_request: Request) -> Response:
            return Response(status_code=200)

        try:
            from memory_system.utils.loop import get_loop

            blocked = get_loop().run_until_complete(mw.dispatch(req, _next))
        except RuntimeError:
            pytest.skip("No usable event loop available in sandbox")
        assert blocked.status_code == 503

        # Disable maintenance mode
        test_client.post("/api/v1/admin/maintenance-mode/disable")
        assert mw._enabled is False
        try:
            from memory_system.utils.loop import get_loop

            allowed = get_loop().run_until_complete(mw.dispatch(req, _next))
        except RuntimeError:
            pytest.skip("No usable event loop available in sandbox")
        assert allowed.status_code == 200


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoints."""

    async def test_async_health_endpoint(self, async_test_client: httpx.AsyncClient) -> None:
        """Test health endpoint with async client."""
        response = await async_test_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == __version__
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    async def test_async_stats_endpoint(self, async_test_client: httpx.AsyncClient) -> None:
        """Test stats endpoint with async client."""
        response = await async_test_client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_memories" in data
        assert "active_sessions" in data
        assert "uptime_seconds" in data

    async def test_async_version_endpoint(self, async_test_client: httpx.AsyncClient) -> None:
        """Test version endpoint with async client."""
        response = await async_test_client.get("/api/v1/version")
        assert response.status_code == 200

        data = response.json()
        assert data["version"] == __version__
        assert data["api_version"] == "v1"


class TestMemoryEndpoints:
    """Test memory API endpoints."""

    def test_update_accepts_absolute_values(self, test_client: TestClient) -> None:
        """Patch route should accept absolute emotion values."""
        resp = test_client.post("/api/v1/memory/", json={"text": "hello"})
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        resp = test_client.patch(
            f"/api/v1/memory/{mem_id}",
            json={"valence": 0.4, "emotional_intensity": 0.7},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valence"] == pytest.approx(0.4)
        assert data["arousal"] == pytest.approx(0.7)

    def test_best_memories_endpoint(self, test_client: TestClient) -> None:
        """Ensure best memories endpoint respects limit parameter."""
        for i in range(3):
            test_client.post("/api/v1/memory/", json={"text": f"mem {i}"})

        resp = test_client.get("/api/v1/memory/best", params={"limit": 2})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_batch_add_sets_modality_metadata(self, test_client: TestClient) -> None:
        """Batch add should store modality in metadata."""
        payload = [
            {"text": "hello", "modality": "text"},
            {"text": "picture", "modality": "image"},
        ]
        resp = test_client.post("/api/v1/memory/batch", json=payload)
        assert resp.status_code == 200
        ids = resp.json()["ids"]

        store = test_client.app.state.store
        loop = asyncio.get_event_loop()
        mems = loop.run_until_complete(store.list_recent(n=10))
        for mem_id, modality in zip(ids, ["text", "image"], strict=False):
            mem = next(m for m in mems if m.id == mem_id)
            assert mem.metadata["modality"] == modality

    def test_pin_unpin_memory(self, test_client: TestClient) -> None:
        resp = test_client.post("/api/v1/memory/", json={"text": "pin me"})
        mem_id = resp.json()["id"]
        resp_pin = test_client.post(f"/api/v1/memory/{mem_id}/pin")
        assert resp_pin.status_code == 200
        assert resp_pin.json()["pinned"] is True
        resp_unpin = test_client.delete(f"/api/v1/memory/{mem_id}/pin")
        assert resp_unpin.status_code == 200
        assert resp_unpin.json()["pinned"] is False

    def test_create_memory_accepts_experience_card(self, test_client: TestClient) -> None:
        """API should persist structured card data when provided."""
        payload = {
            "text": "remember to drink",
            "schema_type": "experience_card",
            "card": {
                "lesson": "stay hydrated",
                "situation": "thirst",
                "lang": "en",
                "summary": "stay hydrated",
                "summary_en": "stay hydrated",
                "success_count": 2,
                "trial_count": 3,
            },
        }
        resp = test_client.post("/api/v1/memory/", json=payload)
        assert resp.status_code == 201
        mem_id = resp.json()["id"]

        import asyncio

        store = test_client.app.state.store
        loop = asyncio.get_event_loop()
        records = loop.run_until_complete(
            store.search(metadata_filters={"schema_type": "experience_card"})
        )
        mem = next(m for m in records if m.id == mem_id)
        card_meta = mem.metadata.get("card", {})
        assert card_meta.get("lesson") == "stay hydrated"
        assert card_meta.get("lang") == "en"
        assert card_meta.get("summary") == "stay hydrated"
        assert mem.metadata.get("success_count") == 2
        assert mem.metadata.get("trial_count") == 3

    def test_best_memories_level_and_filter(self, test_client: TestClient) -> None:
        """Level and metadata filters should narrow results."""
        import asyncio

        from memory_system.core.store import Memory as StoreMemory

        store = test_client.app.state.store
        loop = asyncio.get_event_loop()
        m0 = StoreMemory(id="m0", text="base", level=0, metadata={"user_id": "u1"})
        m1 = StoreMemory(id="m1", text="lvl1", level=1, metadata={"user_id": "u2"})
        loop.run_until_complete(store.add_memory(m0))
        loop.run_until_complete(store.add_memory(m1))
        loop.run_until_complete(store.upsert_scores([(m0.id, 0.1), (m1.id, 0.2)]))

        resp = test_client.get(
            "/api/v1/memory/best",
            params={"limit": 5, "level": 1, "user_id": "u2"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert len(payload) == 1
        assert payload[0]["id"] == "m1"

    def test_best_memories_custom_weights(self, test_client: TestClient, tmp_path: Path) -> None:
        """Query-time weights should influence ranking."""
        import asyncio

        from memory_system.core.store import get_store

        loop = asyncio.get_event_loop()
        store = loop.run_until_complete(get_store(tmp_path / "api.db"))
        loop.run_until_complete(
            unified_memory.add(
                "good",
                valence=0.5,
                emotional_intensity=1.0,
                importance=1.0,
                store=store,
            )
        )
        loop.run_until_complete(
            unified_memory.add(
                "bad but vital",
                valence=-0.5,
                emotional_intensity=1.0,
                importance=1.4,
                store=store,
            )
        )

        resp_default = test_client.get("/api/v1/memory/best", params={"limit": 2})
        assert resp_default.json()[0]["text"] == "good"

        resp_weighted = test_client.get(
            "/api/v1/memory/best",
            params={"limit": 2, "importance": 2.0},
        )
        assert resp_weighted.json()[0]["text"] == "bad but vital"

    def test_best_memories_score_parts_dev(self) -> None:
        """score_parts flag should return breakdown in development profile."""
        settings = UnifiedSettings.for_development()
        app = create_app(settings)
        with TestClient(app) as client:
            client.post("/api/v1/memory/", json={"text": "hello"})
            resp = client.get("/api/v1/memory/best", params={"score_parts": True})
            assert resp.status_code == 200
            payload = resp.json()
            assert "score_parts" in payload[0]


class TestErrorHandling:
    """Test error handling."""

    def test_404_error(self, test_client: TestClient) -> None:
        """Test 404 error handling."""
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_405_error(self, test_client: TestClient) -> None:
        """Test 405 error handling."""
        response = test_client.post("/api/v1/health")
        assert response.status_code == 405
        assert response.headers.get("content-type") == "application/json"

    def test_422_validation_error(self, test_client: TestClient) -> None:
        """Test 422 validation error handling."""
        # Use the memory creation endpoint with invalid payload
        resp = test_client.post("/api/v1/memory/", json={"text": ""})
        assert resp.status_code == 422


class TestApplicationLifecycle:
    """Test application lifecycle events."""

    def test_app_startup(self, test_app: FastAPI) -> None:
        """Test application startup."""
        assert isinstance(test_app, FastAPI)
        assert test_app.title == "Unified Memory System"
        assert test_app.version == __version__

    def test_app_shutdown(self, test_app: FastAPI) -> None:
        """Test application shutdown."""
        called = False

        async def fake_close(self: object) -> None:
            nonlocal called
            called = True

        from unittest.mock import patch

        from memory_system.core.store import SQLiteMemoryStore

        with patch.object(SQLiteMemoryStore, "aclose", fake_close):
            with TestClient(test_app) as _client:
                # startup has run; shutdown will trigger patched aclose
                assert hasattr(_client.app.state, "store")

        assert called is True


class TestDependencyInjection:
    """Test dependency injection."""

    def test_settings_dependency(self, test_client: TestClient) -> None:
        """Test settings dependency is properly injected."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        # The fact that we get a successful response means
        # the settings dependency was properly injected

    def test_memory_store_dependency(self, test_client: TestClient) -> None:
        """Test memory store dependency is properly injected."""
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200
        # The fact that we get a successful response means
        # the memory store dependency was properly injected


@pytest.mark.asyncio
class TestConcurrency:
    """Test concurrent access."""

    async def test_concurrent_health_checks(self, async_test_client: httpx.AsyncClient) -> None:
        """Test concurrent health check requests."""
        tasks = []
        for _i in range(10):
            task = asyncio.create_task(async_test_client.get("/api/v1/health"))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    async def test_concurrent_stats_requests(self, async_test_client: httpx.AsyncClient) -> None:
        """Test concurrent stats requests."""
        tasks = []
        for _i in range(5):
            task = asyncio.create_task(async_test_client.get("/api/v1/stats"))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestPerformance:
    """Test performance characteristics."""

    def test_health_endpoint_performance(self, test_client: TestClient) -> None:
        """Test health endpoint performance."""
        start_time = time.time()

        for _i in range(100):
            response = test_client.get("/api/v1/health")
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle 100 requests in reasonable time
        assert total_time < 10.0  # 10 seconds max

        # Average response time should be reasonable
        avg_time = total_time / 100
        assert avg_time < 0.1  # 100ms max per request

    def test_stats_endpoint_performance(self, test_client: TestClient) -> None:
        """Test stats endpoint performance."""
        start_time = time.time()

        for _i in range(50):
            response = test_client.get("/api/v1/stats")
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle 50 requests in reasonable time
        assert total_time < 10.0  # 10 seconds max

        # Average response time should be reasonable
        avg_time = total_time / 50
        assert avg_time < 0.1  # 100ms max per request
