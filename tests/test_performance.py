"""Performance tests for Unified Memory System."""

import asyncio
import os
import statistics as stats
import time
from collections.abc import AsyncGenerator, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from numpy.typing import NDArray

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from memory_system.core.embedding import EnhancedEmbeddingService
from memory_system.core.enhanced_store import EnhancedMemoryStore
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.store import Memory, SQLiteMemoryStore
from memory_system.core.vector_store import VectorStore
from memory_system.settings import UnifiedSettings
from memory_system.utils.cache import SmartCache
from memory_system.utils.security import (
    EncryptionManager,
    EnhancedPIIFilter,
)

# Default embedding dimension for tests
DIM = UnifiedSettings.for_testing().model.vector_dim

# ---------------------------------------------------------------------------
# Performance thresholds (in milliseconds)
# ---------------------------------------------------------------------------
# These defaults mirror the previously hard-coded values. They can be
# overridden via environment variables when running the tests to accommodate
# slower or faster machines.

MAX_EMBEDDING_TIME_MS = float(os.getenv("MAX_EMBEDDING_TIME_MS", "100"))
MAX_BATCH_PER_TEXT_MS = float(os.getenv("MAX_BATCH_PER_TEXT_MS", "50"))
MAX_BATCH_TOTAL_MS = float(os.getenv("MAX_BATCH_TOTAL_MS", "5000"))
MAX_CONCURRENT_PER_TASK_MS = float(os.getenv("MAX_CONCURRENT_PER_TASK_MS", "200"))
MAX_CONCURRENT_TOTAL_MS = float(os.getenv("MAX_CONCURRENT_TOTAL_MS", "10000"))
EMBEDDING_CACHE_FACTOR = float(os.getenv("EMBEDDING_CACHE_FACTOR", "0.5"))
MAX_EMBEDDING_MEMORY_MB = float(os.getenv("MAX_EMBEDDING_MEMORY_MB", "100"))

MAX_INDEX_AVG_SEARCH_MS = float(os.getenv("MAX_INDEX_AVG_SEARCH_MS", "5"))
MAX_INDEX_MAX_SEARCH_MS = float(os.getenv("MAX_INDEX_MAX_SEARCH_MS", "20"))
MAX_INDEX_BUILD_PER_VECTOR_MS = float(os.getenv("MAX_INDEX_BUILD_PER_VECTOR_MS", "10"))
MAX_INDEX_CONCURRENT_AVG_MS = float(os.getenv("MAX_INDEX_CONCURRENT_AVG_MS", "50"))
MAX_INDEX_CONCURRENT_TOTAL_MS = float(os.getenv("MAX_INDEX_CONCURRENT_TOTAL_MS", "10000"))
MAX_INDEX_MEMORY_KB = float(os.getenv("MAX_INDEX_MEMORY_KB", "10"))

MAX_WRITE_PER_VECTOR_MS = float(os.getenv("MAX_WRITE_PER_VECTOR_MS", "30"))
MAX_FLUSH_TIME_MS = float(os.getenv("MAX_FLUSH_TIME_MS", "5000"))
MAX_READ_AVG_MS = float(os.getenv("MAX_READ_AVG_MS", "1"))
MAX_READ_MAX_MS = float(os.getenv("MAX_READ_MAX_MS", "20"))
MAX_WRITE_CONCURRENT_TOTAL_MS = float(os.getenv("MAX_WRITE_CONCURRENT_TOTAL_MS", "10000"))
MAX_READ_CONCURRENT_TOTAL_MS = float(os.getenv("MAX_READ_CONCURRENT_TOTAL_MS", "10000"))

MAX_CACHE_GET_AVG_MS = float(os.getenv("MAX_CACHE_GET_AVG_MS", "0.1"))
MAX_CACHE_GET_MAX_MS = float(os.getenv("MAX_CACHE_GET_MAX_MS", "1"))
MAX_CACHE_PUT_AVG_MS = float(os.getenv("MAX_CACHE_PUT_AVG_MS", "0.1"))
MAX_CACHE_PUT_MAX_MS = float(os.getenv("MAX_CACHE_PUT_MAX_MS", "5"))
MIN_CACHE_OPS_PER_SEC = float(os.getenv("MIN_CACHE_OPS_PER_SEC", "1000"))
MAX_CACHE_CONCURRENT_TOTAL_MS = float(os.getenv("MAX_CACHE_CONCURRENT_TOTAL_MS", "5000"))

MAX_PII_DETECT_MS = float(os.getenv("MAX_PII_DETECT_MS", "10"))
MAX_PII_REDACT_MS = float(os.getenv("MAX_PII_REDACT_MS", "10"))


@pytest.mark.asyncio
@pytest.mark.perf
class TestEmbeddingPerformance:
    """Test embedding service performance."""

    @pytest_asyncio.fixture
    async def embedding_service(self) -> AsyncGenerator[EnhancedEmbeddingService, None]:
        """Create embedding service for performance tests."""
        settings = UnifiedSettings.for_testing()
        service = EnhancedEmbeddingService(settings.model.model_name, settings)
        yield service
        service.shutdown()

    @pytest.mark.slow
    async def test_single_embedding_performance(
        self, embedding_service: EnhancedEmbeddingService
    ) -> None:
        """Test performance of single text embedding."""
        text = "This is a test sentence for performance evaluation."

        # Warm up
        await embedding_service.embed_text(text)

        # Measure performance
        start_time = time.perf_counter_ns()
        for _ in range(10):
            embedding = await embedding_service.embed_text(text)
            assert embedding.shape[1] == DIM

        end_time = time.perf_counter_ns()
        total_time = (end_time - start_time) / 1e9
        avg_time = total_time / 10

        # Should average less than MAX_EMBEDDING_TIME_MS per embedding
        assert avg_time * 1000 < MAX_EMBEDDING_TIME_MS, f"Average embedding time: {avg_time:.3f}s"

    @pytest.mark.slow
    async def test_batch_embedding_performance(
        self, embedding_service: EnhancedEmbeddingService
    ) -> None:
        """Test performance of batch embedding."""
        texts = [
            f"This is test sentence number {i} for batch performance evaluation." for i in range(50)
        ]

        # Measure batch performance
        start_time = time.perf_counter_ns()
        embeddings = await embedding_service.embed_text(texts)
        end_time = time.perf_counter_ns()

        batch_time = (end_time - start_time) / 1e9
        per_text_time = batch_time / len(texts)

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == DIM

        # Batch processing should be faster per text
        assert per_text_time * 1000 < MAX_BATCH_PER_TEXT_MS, (
            f"Per-text time in batch: {per_text_time:.3f}s"
        )
        assert batch_time * 1000 < MAX_BATCH_TOTAL_MS, f"Total batch time: {batch_time:.3f}s"

    @pytest.mark.slow
    async def test_concurrent_embedding_performance(
        self, embedding_service: EnhancedEmbeddingService
    ) -> None:
        """Test performance under concurrent load."""

        async def embed(text_id: int) -> NDArray[np.float32]:
            text = f"Concurrent test text {text_id}"
            return await embedding_service.embed_text(text)

        # Create concurrent tasks
        num_tasks = 20
        tasks = [embed(i) for i in range(num_tasks)]

        # Measure concurrent performance
        start_time = time.perf_counter_ns()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter_ns()

        concurrent_time = (end_time - start_time) / 1e9
        per_task_time = concurrent_time / num_tasks

        # Verify all results
        assert len(results) == num_tasks
        for embedding in results:
            assert embedding.shape[1] == DIM

        # Should handle concurrent requests efficiently
        assert per_task_time * 1000 < MAX_CONCURRENT_PER_TASK_MS, (
            f"Per-task concurrent time: {per_task_time:.3f}s"
        )
        assert concurrent_time * 1000 < MAX_CONCURRENT_TOTAL_MS, (
            f"Total concurrent time: {concurrent_time:.3f}s"
        )

    async def test_embedding_cache_performance(
        self, embedding_service: EnhancedEmbeddingService
    ) -> None:
        """Test performance improvement from caching."""
        assert 0 < EMBEDDING_CACHE_FACTOR < 1
        text = "This text will be cached for performance testing."

        # Warm up to avoid initialization overhead and populate cache once
        await embedding_service.embed_text(text)

        async def measure(use_cache: bool, n: int = 5) -> float:
            times = []
            for i in range(n):
                query = text if use_cache else f"{text} {i}"
                start = time.perf_counter_ns()
                await embedding_service.embed_text(query)
                times.append((time.perf_counter_ns() - start) / 1e9)
            return stats.median(times)

        first_time = await measure(use_cache=False)
        second_time = await measure(use_cache=True)

        # Results should be identical for cached queries
        cached_embedding = await embedding_service.embed_text(text)
        np.testing.assert_array_equal(cached_embedding, await embedding_service.embed_text(text))

        # Cache hit should provide a noticeable speedup
        # fmt: off
        assert (
            second_time < first_time * EMBEDDING_CACHE_FACTOR
        ), f"Cache hit time: {second_time:.3f}s vs first time: {first_time:.3f}s"
        # fmt: on

    @pytest.mark.slow
    @pytest.mark.skipif(psutil is None, reason="psutil not installed")
    async def test_embedding_memory_usage(
        self, embedding_service: EnhancedEmbeddingService
    ) -> None:
        """Test memory usage during embedding operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate many embeddings
        texts = [f"Memory test text {i}" for i in range(100)]

        for text in texts:
            await embedding_service.embed_text(text)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < MAX_EMBEDDING_MEMORY_MB, (
            f"Memory increased by {memory_increase:.2f}MB"
        )


@pytest.mark.perf
class TestIndexPerformance:
    """Test FAISS index performance."""

    @pytest.fixture
    def large_index(self) -> FaissHNSWIndex:
        """Create index with substantial data."""
        index = FaissHNSWIndex(dim=384)

        # Add 1000 vectors
        num_vectors = 1000
        vector_ids = [f"vec_{i}" for i in range(num_vectors)]
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)

        start_time = time.perf_counter_ns()
        index.add_vectors(vector_ids, vectors)
        build_time = (time.perf_counter_ns() - start_time) / 1e9

        print(f"Index build time for {num_vectors} vectors: {build_time:.3f}s")
        return index

    def test_index_search_performance(self, large_index: FaissHNSWIndex) -> None:
        """Test search performance on large index."""
        query_vector = np.random.rand(384).astype(np.float32)

        # Warm up
        large_index.search(query_vector, k=10)

        # Measure search performance
        search_times = []
        for _ in range(100):
            start_time = time.perf_counter_ns()
            result_ids, distances = large_index.search(query_vector, k=10)
            search_time = (time.perf_counter_ns() - start_time) / 1e9
            search_times.append(search_time)

            assert len(result_ids) <= 10
            assert len(distances) <= 10

        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)

        # Search should be reasonably fast
        assert avg_search_time * 1000 < MAX_INDEX_AVG_SEARCH_MS, (
            f"Average search time: {avg_search_time:.6f}s"
        )
        assert max_search_time * 1000 < MAX_INDEX_MAX_SEARCH_MS, (
            f"Maximum search time: {max_search_time:.6f}s"
        )

    def test_index_build_performance(self) -> None:
        """Test index build performance."""
        index = FaissHNSWIndex(dim=384)

        # Test different batch sizes
        batch_sizes = [100, 500, 1000]

        for batch_size in batch_sizes:
            vector_ids = [f"batch_{batch_size}_vec_{i}" for i in range(batch_size)]
            vectors = np.random.rand(batch_size, 384).astype(np.float32)

            start_time = time.perf_counter_ns()
            index.add_vectors(vector_ids, vectors)
            build_time = (time.perf_counter_ns() - start_time) / 1e9

            per_vector_time = build_time / batch_size

            # Build time should scale reasonably
            # fmt: off
            assert (
                per_vector_time * 1000 < MAX_INDEX_BUILD_PER_VECTOR_MS
            ), f"Per-vector build time: {per_vector_time:.6f}s"
            # fmt: on

    def test_index_concurrent_search(self, large_index: FaissHNSWIndex) -> None:
        """Test concurrent search performance."""

        def search_worker(worker_id: int) -> list[float]:
            """Worker function for concurrent searches."""
            query_vector = np.random.rand(384).astype(np.float32)
            results = []
            for _ in range(20):
                start_time = time.perf_counter_ns()
                result_ids, distances = large_index.search(query_vector, k=5)
                search_time = (time.perf_counter_ns() - start_time) / 1e9
                results.append(search_time)
                assert len(result_ids) <= 5
                assert len(distances) <= 5
            return results

        # Run concurrent searches
        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.perf_counter_ns()
            futures = [executor.submit(search_worker, i) for i in range(num_workers)]
            results = [future.result() for future in futures]
            total_time = (time.perf_counter_ns() - start_time) / 1e9

        # Analyze results
        all_times = []
        for worker_times in results:
            all_times.extend(worker_times)

        avg_search_time = sum(all_times) / len(all_times)
        total_searches = len(all_times)

        # Concurrent searches should maintain good performance
        # fmt: off
        assert (
            avg_search_time * 1000 < MAX_INDEX_CONCURRENT_AVG_MS
        ), f"Average concurrent search time: {avg_search_time:.6f}s"
        # fmt: on
        assert total_time * 1000 < MAX_INDEX_CONCURRENT_TOTAL_MS, (
            f"Total concurrent test time: {total_time:.3f}s"
        )
        print(f"Completed {total_searches} concurrent searches in {total_time:.3f}s")

    @pytest.mark.skipif(psutil is None, reason="psutil not installed")
    def test_index_memory_efficiency(self) -> None:
        """Test index memory efficiency."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create index with many vectors
        index = FaissHNSWIndex(dim=384)
        num_vectors = 2000
        vector_ids = [f"mem_test_vec_{i}" for i in range(num_vectors)]
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)

        index.add_vectors(vector_ids, vectors)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        memory_per_vector = memory_increase / num_vectors * 1024  # KB

        # Memory usage should be reasonable
        assert memory_per_vector < MAX_INDEX_MEMORY_KB, (
            f"Memory per vector: {memory_per_vector:.2f}KB"
        )
        print(f"Index memory usage: {memory_increase:.2f}MB for {num_vectors} vectors")


@pytest.mark.perf
class TestVectorStorePerformance:
    """Test vector store performance."""

    @pytest.fixture
    def vector_store(self, clean_test_vectors: Path) -> Generator[VectorStore, None, None]:
        """Create vector store for performance tests."""
        with VectorStore(clean_test_vectors, dim=384) as store:
            yield store

    def test_vector_store_write_performance(self, vector_store: VectorStore) -> None:
        """Test vector store write performance."""
        num_vectors = 500
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)

        # Test individual writes
        start_time = time.perf_counter_ns()
        for i in range(num_vectors):
            vector_id = f"perf_vec_{i}"
            vector_store.add_vector(vector_id, vectors[i])

        write_time = (time.perf_counter_ns() - start_time) / 1e9
        per_vector_time = write_time / num_vectors

        # Write performance should be reasonable
        assert per_vector_time * 1000 < MAX_WRITE_PER_VECTOR_MS, (
            f"Per-vector write time: {per_vector_time:.6f}s"
        )

        # Test flush performance
        start_time = time.perf_counter_ns()
        try:
            from memory_system.utils.loop import get_loop

            get_loop().run_until_complete(vector_store.flush())
        except RuntimeError:
            pytest.skip("No usable event loop available in sandbox")
        flush_time = (time.perf_counter_ns() - start_time) / 1e9

        assert flush_time * 1000 < MAX_FLUSH_TIME_MS, f"Flush time: {flush_time:.3f}s"

    def test_vector_store_read_performance(self, vector_store: VectorStore) -> None:
        """Test vector store read performance."""
        # Add test vectors
        num_vectors = 100
        vector_ids = []
        for i in range(num_vectors):
            vector_id = f"read_perf_vec_{i}"
            vector = np.random.rand(384).astype(np.float32)
            vector_store.add_vector(vector_id, vector)
            vector_ids.append(vector_id)

        # Test read performance
        read_times = []
        for vector_id in vector_ids:
            start_time = time.perf_counter_ns()
            retrieved = vector_store.get_vector(vector_id)
            read_time = (time.perf_counter_ns() - start_time) / 1e9
            read_times.append(read_time)

            assert retrieved.shape == (384,)

        avg_read_time = sum(read_times) / len(read_times)
        max_read_time = max(read_times)

        # Read performance should be fast
        assert avg_read_time * 1000 < MAX_READ_AVG_MS, f"Average read time: {avg_read_time:.6f}s"
        assert max_read_time * 1000 < MAX_READ_MAX_MS, f"Maximum read time: {max_read_time:.6f}s"

    def test_vector_store_concurrent_access(self, vector_store: VectorStore) -> None:
        """Test concurrent access performance."""

        def write_worker(worker_id: int) -> None:
            """Worker function for concurrent writes."""
            for i in range(50):
                vector_id = f"concurrent_w{worker_id}_v{i}"
                vector = np.random.rand(384).astype(np.float32)
                vector_store.add_vector(vector_id, vector)

        def read_worker(worker_id: int) -> None:
            """Worker function for concurrent reads."""
            # First add some vectors to read
            vector_ids = []
            for i in range(10):
                vector_id = f"concurrent_r{worker_id}_v{i}"
                vector = np.random.rand(384).astype(np.float32)
                vector_store.add_vector(vector_id, vector)
                vector_ids.append(vector_id)

            # Then read them repeatedly
            for _ in range(40):
                for vector_id in vector_ids:
                    vector_store.get_vector(vector_id)

        # Test concurrent writes
        with ThreadPoolExecutor(max_workers=3) as executor:
            start_time = time.perf_counter_ns()
            write_futures = [executor.submit(write_worker, i) for i in range(3)]
            for future in write_futures:
                future.result()
            write_time = (time.perf_counter_ns() - start_time) / 1e9

        assert write_time * 1000 < MAX_WRITE_CONCURRENT_TOTAL_MS, (
            f"Concurrent write time: {write_time:.3f}s"
        )

        # Test concurrent reads
        with ThreadPoolExecutor(max_workers=3) as executor:
            start_time = time.perf_counter_ns()
            read_futures = [executor.submit(read_worker, i) for i in range(3)]
            for future in read_futures:
                future.result()
            read_time = (time.perf_counter_ns() - start_time) / 1e9

        assert read_time * 1000 < MAX_READ_CONCURRENT_TOTAL_MS, (
            f"Concurrent read time: {read_time:.3f}s"
        )


@pytest.mark.perf
class TestCachePerformance:
    """Test cache performance."""

    def test_cache_access_performance(self) -> None:
        """Test cache access performance."""
        cache = SmartCache(max_size=1000, ttl=300)

        # Fill cache
        for i in range(500):
            key = f"key_{i}"
            value = f"value_{i}" * 10  # Some bulk data
            cache.put(key, value)

        # Test get performance
        get_times = []
        for i in range(500):
            key = f"key_{i}"
            start_time = time.perf_counter_ns()
            value = cache.get(key)
            get_time = (time.perf_counter_ns() - start_time) / 1e9
            get_times.append(get_time)

            assert value is not None

        avg_get_time = sum(get_times) / len(get_times)
        max_get_time = max(get_times)

        # Cache access should be very fast
        assert avg_get_time * 1000 < MAX_CACHE_GET_AVG_MS, f"Average get time: {avg_get_time:.8f}s"
        assert max_get_time * 1000 < MAX_CACHE_GET_MAX_MS, f"Maximum get time: {max_get_time:.6f}s"

    def test_cache_put_performance(self) -> None:
        """Test cache put performance."""
        cache = SmartCache(max_size=1000, ttl=300)

        # Test put performance
        put_times = []
        for i in range(500):
            key = f"perf_key_{i}"
            value = f"value_{i}" * 10  # Some bulk data
            start_time = time.perf_counter_ns()
            cache.put(key, value)
            put_time = (time.perf_counter_ns() - start_time) / 1e9
            put_times.append(put_time)

        avg_put_time = sum(put_times) / len(put_times)
        max_put_time = max(put_times)

        # Put operations should be fast
        assert avg_put_time * 1000 < MAX_CACHE_PUT_AVG_MS, f"Average put time: {avg_put_time:.8f}s"
        assert max_put_time * 1000 < MAX_CACHE_PUT_MAX_MS, f"Maximum put time: {max_put_time:.6f}s"

    def test_cache_concurrent_performance(self) -> None:
        """Test cache performance under concurrent load."""
        cache = SmartCache(max_size=5000, ttl=300)

        def worker(worker_id: int) -> None:
            # Each worker does a mix of puts and gets
            for i in range(100):
                key = f"conc_w{worker_id}_k{i}"
                value = f"value_{i}" * 5
                cache.put(key, value)
                # Read some values written by this worker
                if i > 0:
                    prev_key = f"conc_w{worker_id}_k{i - 1}"
                    value = cache.get(prev_key)
                    assert value is not None

        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=4) as executor:
            start_time = time.perf_counter_ns()
            futures = [executor.submit(worker, i) for i in range(4)]
            for future in futures:
                future.result()
            total_time = (time.perf_counter_ns() - start_time) / 1e9

        # Total time should be reasonable for concurrent operations
        assert total_time * 1000 < MAX_CACHE_CONCURRENT_TOTAL_MS, (
            f"Concurrent operation time: {total_time:.3f}s"
        )

        # Calculate operations per second
        total_ops = 4 * 100 * 2  # 4 workers * 100 iterations * (1 put + ~1 get)
        ops_per_sec = total_ops / total_time

        assert ops_per_sec > MIN_CACHE_OPS_PER_SEC, (
            f"Cache operations per second: {ops_per_sec:.1f}"
        )


@pytest.mark.perf
class TestSecurityPerformance:
    """Test security component performance."""

    def test_pii_filter_performance(self) -> None:
        """Test PII filter performance."""
        pii_filter = EnhancedPIIFilter()

        # Generate test texts
        test_texts = [
            f"User {i} with email user{i}@example.com and phone {i:03d}-{i:03d}-{i:04d}"
            for i in range(100)
        ]

        # Test detection performance
        start_time = time.perf_counter_ns()
        for text in test_texts:
            detections = pii_filter.detect(text)
            assert len(detections) >= 2  # email and phone

        detection_time = (time.perf_counter_ns() - start_time) / 1e9
        per_text_time = detection_time / len(test_texts)

        assert per_text_time * 1000 < MAX_PII_DETECT_MS, (
            f"Per-text detection time: {per_text_time:.6f}s"
        )

        # Test redaction performance
        start_time = time.perf_counter_ns()
        for text in test_texts:
            redacted, found_pii, pii_types = pii_filter.redact(text)
            assert found_pii is True
            assert len(pii_types) >= 2

        redaction_time = (time.perf_counter_ns() - start_time) / 1e9
        per_text_time = redaction_time / len(test_texts)

        assert per_text_time * 1000 < MAX_PII_REDACT_MS, (
            f"Per-text redaction time: {per_text_time:.6f}s"
        )

    @pytest.mark.needs_crypto
    def test_encryption_performance(self) -> None:
        """Test encryption performance."""
        encryption_manager = EncryptionManager()

        # Test different data sizes
        test_sizes = [100, 1000, 10000]  # bytes

        for size in test_sizes:
            data = "x" * size

            # Test encryption performance
            encrypted = encryption_manager.encrypt(data)

            # Test decryption performance
            decrypted = encryption_manager.decrypt(encrypted)

            assert decrypted == data


@pytest.mark.asyncio
async def test_dynamic_ef_search_tuning() -> None:
    """Ensure ef_search adapts based on recall measurements."""
    settings = UnifiedSettings.for_testing()
    store = EnhancedMemoryStore(settings)
    await store.start()
    try:
        vec = np.random.rand(settings.model.vector_dim).astype(np.float32)
        mem = await store.add_memory(text="probe", embedding=vec.tolist())

        # Force low recall by expecting a missing id
        store.add_control_query(vec.tolist(), [mem.id, "missing"])
        start_ef = store.vector_store.ef_search
        await store._evaluate_recall()
        assert store.vector_store.ef_search > start_ef

        # Now expect perfect recall and ensure ef_search can decrease
        store._control_queries.clear()
        store.add_control_query(vec.tolist(), [mem.id])
        store.vector_store.set_ef_search(store.vector_store.ef_search * 2)
        high_ef = store.vector_store.ef_search
        await store._evaluate_recall()
        assert store.vector_store.ef_search < high_ef
    finally:
        await store.close()


@pytest.mark.perf
class TestBatchLoadingPerformance:
    """Performance tests for bulk loading operations."""

    @pytest.mark.asyncio
    async def test_store_add_many_bulk(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "bulk.db")
        memories = [Memory.new(f"mem {i}") for i in range(500)]
        start = time.perf_counter_ns()
        await store.add_many(memories)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        per_mem = elapsed / len(memories)
        assert per_mem * 1000 < MAX_WRITE_PER_VECTOR_MS
        await store.aclose()

    def test_index_add_vectors_streaming(self) -> None:
        index = FaissHNSWIndex(dim=384)

        def iterator() -> Iterable[tuple[str, NDArray[np.float32]]]:
            for i in range(500):
                yield f"vec_{i}", np.random.rand(384).astype(np.float32)

        start = time.perf_counter_ns()
        index.add_vectors_streaming(iterator())
        elapsed = (time.perf_counter_ns() - start) / 1e9
        per_vec = elapsed / 500
        assert per_vec * 1000 < MAX_INDEX_BUILD_PER_VECTOR_MS
        assert index.stats().total_vectors == 500
