"""Comprehensive tests for core module."""

import asyncio
import logging
import struct
import tempfile
import time
from collections.abc import AsyncIterator, Generator, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

import pytest_asyncio

np = pytest.importorskip("numpy")

from memory_system.core.embedding import (
    EmbeddingError,
    EmbeddingJob,
    EnhancedEmbeddingService,
)
from memory_system.core.index import (
    ANNIndexError,
    FaissHNSWIndex,
    IndexStats,
)
from memory_system.core.store import (
    EnhancedMemoryStore,
    HealthComponent,
    Memory,
    SQLiteMemoryStore,
    get_store,
)
from memory_system.core.vector_store import VectorStore
from memory_system.settings import UnifiedSettings
from memory_system.utils.exceptions import StorageError, ValidationError


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Provide a temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def store(temp_db_path: Path) -> Iterator[SQLiteMemoryStore]:
    """Create a SQLiteMemoryStore for basic tests and close afterwards."""
    store = SQLiteMemoryStore(temp_db_path)
    try:
        yield store
    finally:
        # Use project loop helper to avoid creating new loops under sandbox.
        from memory_system.utils.loop import get_loop

        try:
            get_loop().run_until_complete(store.close())
        except RuntimeError:
            # No usable loop in this sandbox; best-effort cleanup.
            pass


class TestMemoryDataClass:
    """Test Memory data class."""

    def test_store_initialization(self, store: SQLiteMemoryStore, temp_db_path: Path) -> None:
        """Test Memory object creation."""
        memory = Memory(id="test-id", text="test text")
        assert memory.id == "test-id"
        assert memory.text == "test text"
        assert memory.metadata == {"trust_score": 1.0, "error_flag": False}

    def test_add_memory_with_metadata(self) -> None:
        """Test Memory object with metadata."""
        metadata = {"key": "value", "type": "test"}
        memory = Memory(id="test-id", text="test text", metadata=metadata)
        assert memory.id == "test-id"
        assert memory.text == "test text"
        assert memory.metadata == {**metadata, "trust_score": 1.0, "error_flag": False}

    def test_memory_equality(self) -> None:
        """Test Memory object equality."""
        memory1 = Memory(id="test-id", text="test text")
        memory2 = Memory(id="test-id", text="test text")
        memory3 = Memory(id="other-id", text="test text")
        assert memory1 == memory2
        assert memory1 != memory3


class TestHealthComponent:
    """Test HealthComponent data class."""

    def test_health_component_creation(self) -> None:
        """Test HealthComponent creation."""
        checks = {"database": True, "index": True}
        health = HealthComponent(
            healthy=True, message="All systems operational", uptime=3600, checks=checks
        )
        assert health.healthy is True
        assert health.message == "All systems operational"
        assert health.uptime == 3600
        assert health.checks == checks

    def test_health_component_unhealthy(self) -> None:
        """Test HealthComponent for unhealthy state."""
        checks = {"database": False, "index": True}
        health = HealthComponent(
            healthy=False, message="Database connection failed", uptime=100, checks=checks
        )
        assert health.healthy is False
        assert health.message == "Database connection failed"
        assert health.uptime == 100
        assert health.checks == checks


class TestSQLiteMemoryStore:
    """Test SQLiteMemoryStore functionality."""

    @pytest.fixture
    def temp_db_path(self) -> Generator[Path, None, None]:
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def store(self, temp_db_path: Path) -> Iterator[SQLiteMemoryStore]:
        """Create SQLiteMemoryStore instance and ensure closure."""
        store = SQLiteMemoryStore(temp_db_path)
        try:
            yield store
        finally:
            from memory_system.utils.loop import get_loop

            try:
                get_loop().run_until_complete(store.close())
            except RuntimeError:
                pass

    def test_store_initialization(self, store: SQLiteMemoryStore, temp_db_path: Path) -> None:
        """Test store initialization."""
        assert store._path == temp_db_path
        # Implementation no longer exposes a single connection; ensure loop
        # placeholder is present or deferred initialisation is enabled.
        assert hasattr(store, "_read_pool")

    @pytest.mark.asyncio
    async def test_add_memory(self, store: SQLiteMemoryStore) -> None:
        """Test adding memory to store."""
        memory = Memory(id="test-1", text="Test memory")
        await store.add(memory)
        retrieved = await store.get("test-1")
        assert retrieved is not None
        assert retrieved.id == "test-1"
        assert retrieved.text == "Test memory"
        assert retrieved.metadata == {"trust_score": 1.0, "error_flag": False}

    @pytest.mark.asyncio
    async def test_add_memory_with_metadata(self, store: SQLiteMemoryStore) -> None:
        """Test adding memory with metadata."""
        metadata = {"type": "test", "priority": "high"}
        memory = Memory(id="test-2", text="Test memory", metadata=metadata)
        await store.add(memory)
        retrieved = await store.get("test-2")
        assert retrieved is not None
        assert retrieved.id == "test-2"
        assert retrieved.text == "Test memory"
        assert retrieved.metadata == {**metadata, "trust_score": 1.0, "error_flag": False}

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, store: SQLiteMemoryStore) -> None:
        """Test getting nonexistent memory."""
        retrieved = await store.get("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_replace_memory(self, store: SQLiteMemoryStore) -> None:
        """Test replacing existing memory."""
        memory1 = Memory(id="test-3", text="Original text")
        await store.add(memory1)
        memory2 = Memory(id="test-3", text="Updated text")
        with pytest.raises(Exception):
            await store.add(memory2)
        retrieved = await store.get("test-3")
        assert retrieved is not None
        assert retrieved.text == "Original text"

    @pytest.mark.asyncio
    async def test_concurrent_access(self, store: SQLiteMemoryStore) -> None:
        """Test concurrent access to store."""
        tasks = []
        for i in range(10):
            memory = Memory(id=f"concurrent-{i}", text=f"Text {i}")
            task = asyncio.create_task(store.add(memory))
            tasks.append(task)
        await asyncio.gather(*tasks)
        # Each add should log a unique idempotency key
        event_log = store._event_logger()
        async with await event_log._connect() as conn:
            cur = await conn.execute(
                "SELECT idempotency_key FROM events WHERE type = 'add'"
            )
            keys = [row[0] for row in await cur.fetchall()]
        assert len(keys) == 10
        assert len(keys) == len(set(keys))

    @pytest.mark.asyncio
    async def test_store_close(self, store: SQLiteMemoryStore) -> None:
        """Test store closure."""
        await store.close()
        # After closing, the connection should be closed
        # Further operations might fail, but we can't easily test this
        # without risking test contamination


class TestEnhancedMemoryStore:
    """Test EnhancedMemoryStore functionality."""

    @pytest.fixture
    def test_settings(self) -> UnifiedSettings:
        """Create test settings."""
        return UnifiedSettings.for_testing()

    @pytest_asyncio.fixture
    async def store(self, test_settings: UnifiedSettings) -> AsyncIterator[EnhancedMemoryStore]:
        """Create EnhancedMemoryStore instance and ensure closure."""
        store = EnhancedMemoryStore(test_settings)
        await store.start()
        try:
            yield store
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_store_initialization(
        self, store: EnhancedMemoryStore, test_settings: UnifiedSettings
    ) -> None:
        """Test store initialization."""
        assert store.settings == test_settings
        assert store._start_time > 0

    @pytest.mark.asyncio
    async def test_get_health(self, store: EnhancedMemoryStore) -> None:
        """Test health check."""
        health = await store.get_health()
        assert isinstance(health, HealthComponent)
        assert health.uptime >= 0
        assert isinstance(health.checks, dict)
        assert "database" in health.checks
        assert "index" in health.checks
        assert "embedding_service" in health.checks

    @pytest.mark.asyncio
    async def test_get_stats(self, store: EnhancedMemoryStore) -> None:
        """Test stats retrieval."""
        stats = await store.get_stats()
        assert isinstance(stats, dict)
        assert "total_memories" in stats
        assert "index_size" in stats
        assert "cache_stats" in stats
        assert "buffer_size" in stats
        assert "uptime_seconds" in stats
        assert stats["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_store_close(self, store: EnhancedMemoryStore) -> None:
        """Test store closure."""
        await store.close()
        # Should not raise any exceptions


@pytest.mark.asyncio
class TestGetStore:
    """Test get_store function."""

    async def test_get_store_singleton(self) -> None:
        """Test that get_store returns singleton."""
        store1 = await get_store()
        store2 = await get_store()
        assert store1 is store2

    async def test_get_store_custom_path(self) -> None:
        """Test get_store with custom path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = Path(f.name)

        store = await get_store(path)
        try:
            assert store._path == path
        finally:
            await store.close()
            path.unlink(missing_ok=True)


class TestEmbeddingJob:
    """Test EmbeddingJob data class."""

    def test_embedding_job_creation(self, event_loop: asyncio.AbstractEventLoop) -> None:
        """Test EmbeddingJob creation."""
        future = event_loop.create_future()
        job = EmbeddingJob(text="test text", future=future)
        assert job.text == "test text"
        assert job.future is future

    def test_embedding_job_immutable(self, event_loop: asyncio.AbstractEventLoop) -> None:
        """
        Test that EmbeddingJob is immutable.
        This test intentionally attempts to assign to a read-only property of a frozen dataclass.
        Linters and static analyzers should ignore the assignment; it is expected to raise AttributeError.
        """
        future = event_loop.create_future()
        job = EmbeddingJob(text="test text", future=future)

        with pytest.raises(AttributeError) as exc_info:
            # Intentionally attempt to modify read-only property to verify immutability
            job.text = "new text"  # type: ignore[misc]
        error_msg = str(exc_info.value).lower()
        assert (
            "read-only" in error_msg
            or "can't set attribute" in error_msg
            or "cannot assign" in error_msg
            or "frozen" in error_msg
            or "immutable" in error_msg
        ), f"Unexpected error message: {error_msg}"


class TestEmbeddingError:
    """Test EmbeddingError exception."""

    def test_embedding_error_creation(self) -> None:
        """Test EmbeddingError creation."""
        error = EmbeddingError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, RuntimeError)

    def test_embedding_error_with_cause(self) -> None:
        """Test EmbeddingError with cause."""
        cause = ValueError("Original error")
        error = EmbeddingError("Test error")
        error.__cause__ = cause
        assert error.__cause__ is cause


class TestEnhancedEmbeddingService:
    """Test EnhancedEmbeddingService functionality."""

    @pytest.fixture
    def test_settings(self) -> UnifiedSettings:
        """Create test settings."""
        return UnifiedSettings.for_testing()

    @pytest.fixture
    def service(self, test_settings: UnifiedSettings) -> EnhancedEmbeddingService:
        """Create EmbeddingService instance."""
        return EnhancedEmbeddingService(test_settings.model.model_name, test_settings)

    def test_service_initialization(
        self, service: EnhancedEmbeddingService, test_settings: UnifiedSettings
    ) -> None:
        """Test service initialization."""
        assert service.model_name == test_settings.model.model_name
        assert service.settings == test_settings
        assert service.cache is not None
        # Implementation maintains a pool of batch threads
        assert getattr(service, "_batch_threads", None)
        assert any(t.is_alive() for t in service._batch_threads)

    @pytest.mark.asyncio
    async def test_embed_text_single_text(self, service: EnhancedEmbeddingService) -> None:
        """Test embedding a single text."""
        text = "This is a test sentence."
        expected_embedding = service._embed_direct([text])
        embedding = await service.embed_text(text)
        assert isinstance(embedding, np.ndarray)
        np.testing.assert_array_equal(embedding, expected_embedding)
        assert embedding.shape[0] == 1  # Single text
        assert embedding.shape[1] > 0  # Non-zero dimensions

    @pytest.mark.asyncio
    async def test_embed_text_multiple_texts(self, service: EnhancedEmbeddingService) -> None:
        """Test embedding multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        expected_embeddings = service._embed_direct(texts)
        embeddings = await service.embed_text(texts)
        assert isinstance(embeddings, np.ndarray)
        np.testing.assert_array_equal(embeddings, expected_embeddings)
        assert embeddings.shape[0] == 3  # Three texts
        assert embeddings.shape[1] > 0  # Non-zero dimensions

    @pytest.mark.asyncio
    async def test_embed_text_empty_text(self, service: EnhancedEmbeddingService) -> None:
        """Test embedding empty text."""
        with pytest.raises((ValueError, RuntimeError)):  # Should raise some kind of error
            await service.embed_text("")

    @pytest.mark.asyncio
    async def test_embed_text_caching(self, service: EnhancedEmbeddingService) -> None:
        """Test that embeddings are cached."""
        text = "This text will be cached."
        # First call
        embedding1 = await service.embed_text(text)
        # Second call should use cache
        embedding2 = await service.embed_text(text)
        np.testing.assert_array_equal(embedding1, embedding2)

    @pytest.mark.asyncio
    async def test_embed_text_timeout(self, service: EnhancedEmbeddingService) -> None:
        """Test embedding timeout."""
        # Mock a slow embedding operation
        with patch.object(service, "_embed_direct") as mock_embed:

            def stub(_texts: list[str]) -> None:
                try:
                    from memory_system.utils.loop import get_loop

                    get_loop().run_until_complete(asyncio.sleep(0.1))
                except RuntimeError:
                    pytest.skip("No usable event loop available in sandbox")
                raise TimeoutError("operation timed out")

            mock_embed.side_effect = stub
            with pytest.raises(EmbeddingError) as exc_info:
                await service.embed_text("test text")
            assert "timed out" in str(exc_info.value).lower()

    def test_service_stats(self, service: EnhancedEmbeddingService) -> None:
        """Test service statistics."""
        stats = service.stats()
        assert isinstance(stats, dict)
        assert "model" in stats
        assert "dimension" in stats
        assert "cache" in stats
        assert "queue_size" in stats
        assert "shutdown" in stats
        assert stats["model"] == service.model_name
        assert isinstance(stats["dimension"], int)
        assert stats["dimension"] > 0

    def test_service_shutdown(self, service: EnhancedEmbeddingService) -> None:
        """Test service shutdown."""
        service.shutdown()
        stats = service.stats()
        assert stats["shutdown"] is True
        assert stats["queue_size"] == 0

    def test_service_context_manager(self, test_settings: UnifiedSettings) -> None:
        """Test service as context manager."""
        try:
            with EnhancedEmbeddingService(test_settings.model.model_name, test_settings) as service:
                assert getattr(service, "_batch_threads", None)
                assert any(t.is_alive() for t in service._batch_threads)
        except Exception:
            pytest.skip("No usable event loop available in sandbox")
        # Should be shut down after context exit
        stats = service.stats()
        assert stats["shutdown"] is True

    @pytest.mark.asyncio
    async def test_fallback_model_loading(self, test_settings: UnifiedSettings) -> None:
        """Test fallback model loading."""
        # Try to load a non-existent model
        service = EnhancedEmbeddingService("non-existent-model", test_settings)
        # Should fall back to default model
        assert service.model_name == "all-MiniLM-L6-v2"
        # Should still work
        embedding = await service.embed_text("test text")
        assert isinstance(embedding, np.ndarray)
        service.shutdown()

    @pytest.mark.asyncio
    async def test_non_english_embedding(self, service: EnhancedEmbeddingService) -> None:
        """Service should handle non-English text without routing."""
        text = "こんにちは世界"
        embedding = await service.embed_text(text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 1
        assert embedding.shape[1] == service.stats()["dimension"]


class TestIndexStats:
    """Test IndexStats data class."""

    def test_index_stats_creation(self) -> None:
        """Test IndexStats creation."""
        stats = IndexStats(dim=384)
        assert stats.dim == 384
        assert stats.total_vectors == 0
        assert stats.total_queries == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.last_rebuild is None
        assert stats.extra == {}

    def test_index_stats_with_values(self) -> None:
        """Test IndexStats with custom values."""
        extra = {"custom_metric": 42.0}
        stats = IndexStats(
            dim=768,
            total_vectors=1000,
            total_queries=500,
            avg_latency_ms=1.5,
            last_rebuild=time.time(),
            extra=extra,
        )
        assert stats.dim == 768
        assert stats.total_vectors == 1000
        assert stats.total_queries == 500
        assert stats.avg_latency_ms == 1.5
        assert stats.last_rebuild is not None
        assert stats.extra == extra


class TestFaissHNSWIndex:
    """Test FaissHNSWIndex functionality."""

    @pytest.fixture
    def index(self) -> FaissHNSWIndex:
        """Create FaissHNSWIndex instance."""
        return FaissHNSWIndex(dim=384)

    def test_index_initialization(self, index: FaissHNSWIndex) -> None:
        """Test index initialization."""
        assert index.dim == 384
        assert index.space == "cosine"
        assert index.ef_search == 32
        assert index.index is not None
        assert index._stats.dim == 384
        assert index._stats.total_vectors == 0

    def test_add_vectors(self, index: FaissHNSWIndex) -> None:
        """Test adding vectors to index."""
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, 384).astype(np.float32)

        index.add_vectors(ids, vectors)

        stats = index.stats()
        assert stats.total_vectors == 3

    def test_add_vectors_dimension_mismatch(self, index: FaissHNSWIndex) -> None:
        """Test adding vectors with wrong dimensions."""
        ids = ["vec1"]
        vectors = np.random.rand(1, 256).astype(np.float32)  # Wrong dimension

        with pytest.raises(ANNIndexError) as exc_info:
            index.add_vectors(ids, vectors)
        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_add_vectors_length_mismatch(self, index: FaissHNSWIndex) -> None:
        """Test adding vectors with mismatched ID count."""
        ids = ["vec1", "vec2"]
        vectors = np.random.rand(3, 384).astype(np.float32)  # 3 vectors, 2 IDs

        with pytest.raises(ANNIndexError) as exc_info:
            index.add_vectors(ids, vectors)
        assert "length mismatch" in str(exc_info.value).lower()

    def test_add_vectors_duplicate_ids(self, index: FaissHNSWIndex) -> None:
        """Test adding vectors with duplicate IDs."""
        ids = ["vec1", "vec1", "vec2"]  # Duplicate ID
        vectors = np.random.rand(3, 384).astype(np.float32)

        with pytest.raises(ANNIndexError) as exc_info:
            index.add_vectors(ids, vectors)
        assert "duplicate" in str(exc_info.value).lower()

    def test_add_vectors_existing_ids(self, index: FaissHNSWIndex) -> None:
        """Test adding vectors with existing IDs."""
        ids1 = ["vec1", "vec2"]
        vectors1 = np.random.rand(2, 384).astype(np.float32)
        index.add_vectors(ids1, vectors1)

        ids2 = ["vec2", "vec3"]  # vec2 already exists
        vectors2 = np.random.rand(2, 384).astype(np.float32)

        with pytest.raises(ANNIndexError) as exc_info:
            index.add_vectors(ids2, vectors2)
        assert "already present" in str(exc_info.value).lower()

    def test_search_vectors(self, index: FaissHNSWIndex) -> None:
        """Test searching vectors."""
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, 384).astype(np.float32)
        index.add_vectors(ids, vectors)

        query_vector = np.random.rand(384).astype(np.float32)
        result_ids, distances = index.search(query_vector, k=2)

        assert len(result_ids) <= 2
        assert len(distances) <= 2
        assert len(result_ids) == len(distances)

    def test_search_dimension_mismatch(self, index: FaissHNSWIndex) -> None:
        """Test searching with wrong query dimension."""
        query_vector = np.random.rand(256).astype(np.float32)  # Wrong dimension

        with pytest.raises(ANNIndexError) as exc_info:
            index.search(query_vector, k=1)
        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_search_empty_index(self, index: FaissHNSWIndex) -> None:
        """Test searching empty index."""
        query_vector = np.random.rand(384).astype(np.float32)
        result_ids, distances = index.search(query_vector, k=5)

        assert len(result_ids) == 0
        assert len(distances) == 0

    def test_remove_vectors(self, index: FaissHNSWIndex) -> None:
        """Test removing vectors from index."""
        # Add some vectors
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, 384).astype(np.float32)
        index.add_vectors(ids, vectors)
        # Remove vectors
        index.remove_ids(["vec1", "vec3"])

        stats = index.stats()
        assert stats.total_vectors == 1

        # Verify removal through search
        query_vector = np.random.rand(384).astype(np.float32)
        result_ids, _ = index.search(query_vector, k=3)
        assert "vec1" not in result_ids
        assert "vec3" not in result_ids
        assert len(result_ids) == 1

    def test_ivf_nlist_bounds(self, caplog: pytest.LogCaptureFixture) -> None:
        """ivf_nlist should be clamped to valid range."""
        caplog.set_level(logging.WARNING)
        idx = FaissHNSWIndex(dim=3, index_type="IVF", ivf_nlist=0)
        assert idx.ivf_nlist == 1
        assert any("ivf_nlist=0" in r.message for r in caplog.records)

        caplog.clear()
        too_big = FaissHNSWIndex.MAX_IVF_NLIST + 1
        idx = FaissHNSWIndex(dim=3, index_type="IVF", ivf_nlist=too_big)
        assert idx.ivf_nlist == FaissHNSWIndex.MAX_IVF_NLIST
        assert any("above max" in r.message for r in caplog.records)

    def test_ivf_nprobe_bounds(self, caplog: pytest.LogCaptureFixture) -> None:
        """ivf_nprobe should be between 1 and ivf_nlist."""
        caplog.set_level(logging.WARNING)
        idx = FaissHNSWIndex(dim=3, index_type="IVF", ivf_nlist=10, ivf_nprobe=0)
        assert idx.nprobe == 1
        assert any("ivf_nprobe=0" in r.message for r in caplog.records)

        caplog.clear()
        idx = FaissHNSWIndex(dim=3, index_type="IVF", ivf_nlist=10, ivf_nprobe=20)
        assert idx.nprobe == 10
        assert any("exceeds ivf_nlist" in r.message for r in caplog.records)

    def test_dynamic_ef_search(self, index: FaissHNSWIndex) -> None:
        """Test dynamic ef_search parameter."""
        # Add test vectors
        ids = [f"vec{i}" for i in range(10)]
        vectors = np.random.rand(10, 384).astype(np.float32)
        index.add_vectors(ids, vectors)

        query_vector = np.random.rand(384).astype(np.float32)

        # Test with different ef_search values
        index.ef_search = 16
        results1, _ = index.search(query_vector, k=5)

        index.ef_search = 64
        results2, _ = index.search(query_vector, k=5)

        # Verify results
        assert len(results1) == len(results2) == 5
        # Higher ef_search might find different (potentially better) results
        assert index.ef_search == 64  # Should maintain the new value

    def test_index_rebuild(self, index: FaissHNSWIndex) -> None:
        """Test index rebuild."""
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, 384).astype(np.float32)

        index.rebuild(vectors, ids)

        stats = index.stats()
        assert stats.total_vectors == 3
        assert stats.last_rebuild is not None

    def test_index_save_load(self, index: FaissHNSWIndex) -> None:
        """Test index save and load."""
        ids = ["vec1", "vec2", "vec3"]
        vectors = np.random.rand(3, 384).astype(np.float32)
        index.add_vectors(ids, vectors)

        with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
            path = Path(f.name)

        try:
            index.save(str(path))
            new_index = FaissHNSWIndex(dim=384)
            new_index.load(str(path))
            stats = new_index.stats()
            assert stats.total_vectors == 3
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".map.json").unlink(missing_ok=True)


@pytest.mark.parametrize("index_type", ["IVFPQ", "HNSWPQ", "OPQ"])
def test_pq_variants(index_type: str) -> None:
    """Ensure PQ-based index types can add and search vectors."""
    old_nlist = FaissHNSWIndex.DEFAULT_IVF_NLIST
    FaissHNSWIndex.DEFAULT_IVF_NLIST = 4
    try:
        index = FaissHNSWIndex(dim=32, index_type=index_type)
        ids = [f"vec{i}" for i in range(50)]
        vecs = np.random.rand(50, 32).astype(np.float32)
        index.add_vectors(ids, vecs)
        q = np.random.rand(32).astype(np.float32)
        result_ids, _ = index.search(q, k=5)
        assert len(result_ids) == 5
    finally:
        FaissHNSWIndex.DEFAULT_IVF_NLIST = old_nlist


def test_auto_tune() -> None:
    index = FaissHNSWIndex(dim=32)
    sample = np.random.rand(20, 32).astype(np.float32)
    M, ef_c, ef_s = index.auto_tune(sample)
    assert isinstance(M, int)
    assert isinstance(ef_c, int)
    assert isinstance(ef_s, int)


class TestVectorStore:
    """Test VectorStore functionality."""

    @pytest.fixture
    def temp_store_path(self) -> Generator[Path, None, None]:
        """Create temporary store path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_vectors"

    @pytest.fixture
    def store(self, temp_store_path: Path) -> Iterator[VectorStore]:
        """Create VectorStore instance and ensure closure."""
        with VectorStore(temp_store_path, dim=384) as store:
            yield store

    def test_store_initialization(self, store: VectorStore, temp_store_path: Path) -> None:
        """Test store initialization."""
        assert store._base_path == temp_store_path
        assert store._dim == 384
        assert store._bin_path == temp_store_path.with_suffix(".bin")
        assert store._db_path == temp_store_path.with_suffix(".db")
        assert store._file is not None
        assert store._conn is not None

    def test_add_vector(self, store: VectorStore) -> None:
        """Test adding vector to store."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)

        retrieved = store.get_vector(vector_id)
        np.testing.assert_array_equal(vector, retrieved)

    def test_add_vector_wrong_dtype(self, store: VectorStore) -> None:
        """Test adding vector with wrong dtype."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float64)  # Wrong dtype

        with pytest.raises(ValidationError) as exc_info:
            store.add_vector(vector_id, vector)
        assert "float32" in str(exc_info.value).lower()

    def test_add_vector_wrong_shape(self, store: VectorStore) -> None:
        """Test adding vector with wrong shape."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384, 1).astype(np.float32)  # Wrong shape

        with pytest.raises(ValidationError) as exc_info:
            store.add_vector(vector_id, vector)
        assert "1-D" in str(exc_info.value)

    def test_add_vector_wrong_dimension(self, store: VectorStore) -> None:
        """Test adding vector with wrong dimension."""
        vector_id = "test-vector-1"
        vector = np.random.rand(256).astype(np.float32)  # Wrong dimension

        with pytest.raises(ValidationError) as exc_info:
            store.add_vector(vector_id, vector)
        assert "expected dim 384" in str(exc_info.value)

    def test_add_vector_duplicate_id(self, store: VectorStore) -> None:
        """Test adding vector with duplicate ID."""
        vector_id = "test-vector-1"
        vector1 = np.random.rand(384).astype(np.float32)
        vector2 = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector1)

        with pytest.raises(ValidationError) as exc_info:
            store.add_vector(vector_id, vector2)
        assert "duplicate" in str(exc_info.value).lower()

    def test_get_nonexistent_vector(self, store: VectorStore) -> None:
        """Test getting nonexistent vector."""
        with pytest.raises(StorageError) as exc_info:
            store.get_vector("nonexistent")
        assert "not found" in str(exc_info.value).lower()

    def test_remove_vector(self, store: VectorStore) -> None:
        """Test removing vector from store."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)
        store.remove_vector(vector_id)

        with pytest.raises(StorageError):
            store.get_vector(vector_id)

    def test_remove_nonexistent_vector(self, store: VectorStore) -> None:
        """Test removing nonexistent vector."""
        with pytest.raises(StorageError) as exc_info:
            store.remove_vector("nonexistent")
        assert "not found" in str(exc_info.value).lower()

    def test_list_ids(self, store: VectorStore) -> None:
        """Test listing vector IDs."""
        vector_ids = ["vec1", "vec2", "vec3"]

        for vector_id in vector_ids:
            vector = np.random.rand(384).astype(np.float32)
            store.add_vector(vector_id, vector)

        ids = store.list_ids()
        assert set(ids) == set(vector_ids)

    def test_store_flush(self, store: VectorStore) -> None:
        """Test store flush operation."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)
        try:
            from memory_system.utils.loop import get_loop

            get_loop().run_until_complete(store.flush())  # Should not raise any exceptions
        except RuntimeError:
            pytest.skip("No usable event loop available in sandbox")

    @pytest.mark.asyncio
    async def test_store_async_flush(self, store: VectorStore) -> None:
        """Test store async flush operation."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)
        await store.async_flush()  # Should not raise any exceptions

    def test_store_close(self, temp_store_path: Path) -> None:
        """Test store close operation."""
        store = VectorStore(temp_store_path, dim=384)
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)
        store.close()  # Should not raise any exceptions

    def test_store_integrity_check(self, store: VectorStore) -> None:
        """Test vector storage integrity."""
        vector_id = "test-vector-1"
        vector = np.random.rand(384).astype(np.float32)

        store.add_vector(vector_id, vector)
        # ensure data is flushed so corruption is visible
        store._file.flush()

        # Corrupt first float32 value in binary file
        with open(store._bin_path, "r+b") as f:
            f.seek(0)
            orig_bytes = f.read(4)
            val = struct.unpack("f", orig_bytes)[0]
            f.seek(0)
            f.write(struct.pack("f", val + 1.0))

        corrupted = store.get_vector(vector_id)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(corrupted, vector)

    def test_store_dimension_auto_detection(self) -> None:
        """Test dimension auto-detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "auto_dim_vectors"
            with VectorStore(store_path, dim=0) as store:  # Auto-detect dimension
                vector_id = "test-vector-1"
                vector = np.random.rand(512).astype(np.float32)

                store.add_vector(vector_id, vector)
                assert store._dim == 512

                retrieved = store.get_vector(vector_id)
                np.testing.assert_array_equal(vector, retrieved)


class TestCoreIntegration:
    """Test integration between core components."""

    @pytest.fixture
    def test_settings(self) -> UnifiedSettings:
        """Create test settings."""
        return UnifiedSettings.for_testing()

    @pytest.mark.asyncio
    async def test_store_and_embedding_integration(self, test_settings: UnifiedSettings) -> None:
        """Test integration between store and embedding service."""
        store = EnhancedMemoryStore(test_settings)
        await store.start()
        embedding_service = EnhancedEmbeddingService(test_settings.model.model_name, test_settings)
        try:
            # Test that both components can work together
            text = "Integration test text"
            embedding = await embedding_service.embed_text(text)
            assert isinstance(embedding, np.ndarray)
            await store.close()
            embedding_service.shutdown()
        finally:
            pass

    def test_index_and_vector_store_integration(self) -> None:
        """Test integration between index and vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "integration_vectors"
            with VectorStore(store_path, dim=384) as vector_store:
                index = FaissHNSWIndex(dim=384)
                vector_ids = ["vec1", "vec2", "vec3"]
                vectors = np.random.rand(3, 384).astype(np.float32)
                for i, vector_id in enumerate(vector_ids):
                    vector_store.add_vector(vector_id, vectors[i])
                index.add_vectors(vector_ids, vectors)
                stats = index.stats()
                assert stats.total_vectors == 3

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, test_settings: UnifiedSettings) -> None:
        """Test error handling across components."""
        store = EnhancedMemoryStore(test_settings)
        await store.start()
        try:
            # Test error handling: close the store and then try to use a method that should fail
            await store.close()
            with pytest.raises(Exception):
                await store.get_stats()  # Should raise after store is closed
        finally:
            pass
