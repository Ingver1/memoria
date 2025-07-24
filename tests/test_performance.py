"""Performance tests for Unified Memory System."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import numpy as np
import psutil
import pytest_asyncio
from memory_system.config.settings import UnifiedSettings
from memory_system.core.embedding import EnhancedEmbeddingService
from memory_system.core.index import FaissHNSWIndex
from memory_system.core.vector_store import VectorStore
from memory_system.utils.cache import SmartCache
from memory_system.utils.security import (
    EncryptionManager,
    EnhancedPIIFilter,
)


@pytest.mark.asyncio
@pytest.mark.performance
class TestEmbeddingPerformance:
    """Test embedding service performance."""

    @pytest_asyncio.fixture
    async def embedding_service(self):
        """Create embedding service for performance tests."""
        settings = UnifiedSettings.for_testing()
        service = EnhancedEmbeddingService("all-MiniLM-L6-v2", settings)
        yield service
        service.shutdown()

    @pytest.mark.slow
    async def test_single_embedding_performance(self, embedding_service):
        """Test performance of single text embedding."""
        text = "This is a test sentence for performance evaluation."

        # Warm up
        await embedding_service.encode(text)

        # Measure performance
        start_time = time.time()
        for _ in range(10):
            embedding = await embedding_service.encode(text)
            assert embedding.shape[1] == 384

        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 10

        # Should average less than 100ms per embedding (with caching)
        assert avg_time < 0.1, f"Average embedding time: {avg_time:.3f}s"

    @pytest.mark.slow
    async def test_batch_embedding_performance(self, embedding_service):
        """Test performance of batch embedding."""
        texts = [
            f"This is test sentence number {i} for batch performance evaluation." for i in range(50)
        ]

        # Measure batch performance
        start_time = time.time()
        embeddings = await embedding_service.encode(texts)
        end_time = time.time()

        batch_time = end_time - start_time
        per_text_time = batch_time / len(texts)

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384

        # Batch processing should be faster per text
        assert per_text_time < 0.05, f"Per-text time in batch: {per_text_time:.3f}s"
        assert batch_time < 5.0, f"Total batch time: {batch_time:.3f}s"

    @pytest.mark.slow
    async def test_concurrent_embedding_performance(self, embedding_service):
        """Test performance under concurrent load."""

        async def embed_text(text_id: int):
            text = f"Concurrent test text {text_id}"
            return await embedding_service.encode(text)

        # Create concurrent tasks
        num_tasks = 20
        tasks = [embed_text(i) for i in range(num_tasks)]

        # Measure concurrent performance
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        concurrent_time = end_time - start_time
        per_task_time = concurrent_time / num_tasks

        # Verify all results
        assert len(results) == num_tasks
        for embedding in results:
            assert embedding.shape[1] == 384

        # Should handle concurrent requests efficiently
        assert per_task_time < 0.2, f"Per-task concurrent time: {per_task_time:.3f}s"
        assert concurrent_time < 10.0, f"Total concurrent time: {concurrent_time:.3f}s"

    async def test_embedding_cache_performance(self, embedding_service):
        """Test performance improvement from caching."""
        text = "This text will be cached for performance testing."

        # First embedding (cache miss)
        start_time = time.time()
        embedding1 = await embedding_service.encode(text)
        first_time = time.time() - start_time

        # Second embedding (cache hit)
        start_time = time.time()
        embedding2 = await embedding_service.encode(text)
        second_time = time.time() - start_time

        # Results should be identical
        np.testing.assert_array_equal(embedding1, embedding2)

        # Cache hit should be significantly faster
        assert second_time < first_time * 0.1, (
            f"Cache hit time: {second_time:.3f}s vs first time: {first_time:.3f}s"
        )

    @pytest.mark.slow
    async def test_embedding_memory_usage(self, embedding_service):
        """Test memory usage during embedding operations."""

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Generate many embeddings
        texts = [f"Memory test text {i}" for i in range(100)]

        for text in texts:
            await embedding_service.encode(text)

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"


@pytest.mark.performance
class TestIndexPerformance:
    """Test FAISS index performance."""

    @pytest.fixture
    def large_index(self):
        """Create index with substantial data."""
        index = FaissHNSWIndex(dim=384)

        # Add 1000 vectors
        num_vectors = 1000
        vector_ids = [f"vec_{i}" for i in range(num_vectors)]
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)

        start_time = time.time()
        index.add_vectors(vector_ids, vectors)
        build_time = time.time() - start_time

        print(f"Index build time for {num_vectors} vectors: {build_time:.3f}s")
        return index

    def test_index_search_performance(self, large_index):
        """Test search performance on large index."""
        query_vector = np.random.rand(384).astype(np.float32)

        # Warm up
        large_index.search(query_vector, k=10)

        # Measure search performance
        search_times = []
        for _ in range(100):
            start_time = time.time()
            result_ids, distances = large_index.search(query_vector, k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)

            assert len(result_ids) <= 10
            assert len(distances) <= 10

        avg_search_time = sum(search_times) / len(search_times)
        max_search_time = max(search_times)

        # Search should be fast
        assert avg_search_time < 0.001, f"Average search time: {avg_search_time:.6f}s"
        assert max_search_time < 0.011, f"Maximum search time: {max_search_time:.6f}s"

    def test_index_build_performance(self):
        """Test index build performance."""
        index = FaissHNSWIndex(dim=384)

        # Test different batch sizes
        batch_sizes = [100, 500, 1000]

        for batch_size in batch_sizes:
            vector_ids = [f"batch_{batch_size}_vec_{i}" for i in range(batch_size)]
            vectors = np.random.rand(batch_size, 384).astype(np.float32)

            start_time = time.time()
            index.add_vectors(vector_ids, vectors)
            build_time = time.time() - start_time

            per_vector_time = build_time / batch_size

            # Build time should scale reasonably
            assert per_vector_time < 0.01, f"Per-vector build time: {per_vector_time:.6f}s"

    def test_index_concurrent_search(self, large_index):
        """Test concurrent search performance."""

        def search_worker(worker_id: int):
            """Worker function for concurrent searches."""
            query_vector = np.random.rand(384).astype(np.float32)
            results = []

            for _ in range(20):
                start_time = time.time()
                result_ids, distances = large_index.search(query_vector, k=5)
                search_time = time.time() - start_time
                results.append(search_time)

                assert len(result_ids) <= 5
                assert len(distances) <= 5

            return results

        # Run concurrent searches
        num_workers = 4
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.time()
            futures = [executor.submit(search_worker, i) for i in range(num_workers)]
            results = [future.result() for future in futures]
            total_time = time.time() - start_time

        # Analyze results
        all_times = []
        for worker_times in results:
            all_times.extend(worker_times)

        avg_search_time = sum(all_times) / len(all_times)
        total_searches = len(all_times)

        # Concurrent searches should maintain good performance
        assert avg_search_time < 0.03, f"Average concurrent search time: {avg_search_time:.6f}s"
        assert total_time < 10.0, f"Total concurrent test time: {total_time:.3f}s"
        print(f"Completed {total_searches} concurrent searches in {total_time:.3f}s")

    def test_index_memory_efficiency(self):
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
        memory_per_vector = memory_increase / num_vectors * 1024  # KB per vector

        # Memory usage should be reasonable
        assert memory_per_vector < 10, f"Memory per vector: {memory_per_vector:.2f}KB"
        print(f"Index memory usage: {memory_increase:.2f}MB for {num_vectors} vectors")


@pytest.mark.performance
class TestVectorStorePerformance:
    """Test vector store performance."""

    @pytest.fixture
    def vector_store(self, clean_test_vectors):
        """Create vector store for performance tests."""
        store = VectorStore(clean_test_vectors, dim=384)
        yield store
        store.close()

    def test_vector_store_write_performance(self, vector_store):
        """Test vector store write performance."""
        num_vectors = 500
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)

        # Test individual writes
        start_time = time.time()
        for i in range(num_vectors):
            vector_id = f"perf_vec_{i}"
            vector_store.add_vector(vector_id, vectors[i])

        write_time = time.time() - start_time
        per_vector_time = write_time / num_vectors

        # Write performance should be reasonable
        assert per_vector_time < 0.01, f"Per-vector write time: {per_vector_time:.6f}s"

        # Test flush performance
        start_time = time.time()
        asyncio.run(vector_store.flush())
        flush_time = time.time() - start_time

        assert flush_time < 5.0, f"Flush time: {flush_time:.3f}s"

    def test_vector_store_read_performance(self, vector_store):
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
            start_time = time.time()
            retrieved = vector_store.get_vector(vector_id)
            read_time = time.time() - start_time
            read_times.append(read_time)

            assert retrieved.shape == (384,)

        avg_read_time = sum(read_times) / len(read_times)
        max_read_time = max(read_times)

        # Read performance should be fast
        assert avg_read_time < 0.001, f"Average read time: {avg_read_time:.6f}s"
        assert max_read_time < 0.01, f"Maximum read time: {max_read_time:.6f}s"

    def test_vector_store_concurrent_access(self, vector_store):
        """Test concurrent access performance."""

        def write_worker(worker_id: int):
            """Worker function for concurrent writes."""
            for i in range(50):
                vector_id = f"concurrent_w{worker_id}_v{i}"
                vector = np.random.rand(384).astype(np.float32)
                vector_store.add_vector(vector_id, vector)

        def read_worker(worker_id: int):
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
            start_time = time.time()
            write_futures = [executor.submit(write_worker, i) for i in range(3)]
            for future in write_futures:
                future.result()
            write_time = time.time() - start_time

        assert write_time < 10.0, f"Concurrent write time: {write_time:.3f}s"

        # Test concurrent reads
        with ThreadPoolExecutor(max_workers=3) as executor:
            start_time = time.time()
            read_futures = [executor.submit(read_worker, i) for i in range(3)]
            for future in read_futures:
                future.result()
            read_time = time.time() - start_time

        assert read_time < 10.0, f"Concurrent read time: {read_time:.3f}s"


@pytest.mark.performance
class TestCachePerformance:
    """Test cache performance."""

    def test_cache_access_performance(self):
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
            start_time = time.time()
            value = cache.get(key)
            get_time = time.time() - start_time
            get_times.append(get_time)

            assert value is not None

        avg_get_time = sum(get_times) / len(get_times)
        max_get_time = max(get_times)

        # Cache access should be very fast
        assert avg_get_time < 0.0001, f"Average get time: {avg_get_time:.8f}s"
        assert max_get_time < 0.001, f"Maximum get time: {max_get_time:.6f}s"

    def test_cache_put_performance(self):
        """Test cache put performance."""
        cache = SmartCache(max_size=1000, ttl=300)

        # Test put performance
        put_times = []
        for i in range(500):
            key = f"key_{i}"
            value = f"value_{i}" * 10  # Some bulk data

            start_time = time.time()
            cache.put(key, value)
            put_time = time.time() - start_time
            put_times.append(put_time)

        avg_put_time = sum(put_times) / len(put_times)
        max_put_time = max(put_times)

        # Cache put should be fast
        assert avg_put_time < 0.0001, f"Average put time: {avg_put_time:.8f}s"
        assert max_put_time < 0.005, f"Maximum put time: {max_put_time:.6f}s"

    def test_cache_concurrent_performance(self):
        """Test cache performance under concurrent access."""
        cache = SmartCache(max_size=1000, ttl=300)

        def cache_worker(worker_id: int):
            """Worker function for concurrent cache access."""
            for i in range(100):
                key = f"worker_{worker_id}_key_{i}"
                value = f"worker_{worker_id}_value_{i}"

                # Put and get
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value

        # Run concurrent workers
        num_workers = 5
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            start_time = time.time()
            futures = [executor.submit(cache_worker, i) for i in range(num_workers)]
            for future in futures:
                future.result()
            total_time = time.time() - start_time

        operations_per_second = (num_workers * 100 * 2) / total_time  # 2 ops per iteration

        # Should handle many concurrent operations
        assert operations_per_second > 1000, f"Operations per second: {operations_per_second:.0f}"
        assert total_time < 5.0, f"Total concurrent time: {total_time:.3f}s"


@pytest.mark.performance
class TestSecurityPerformance:
    """Test security component performance."""

    def test_pii_filter_performance(self):
        """Test PII filter performance."""
        pii_filter = EnhancedPIIFilter()

        # Generate test texts
        test_texts = [
            f"User {i} with email user{i}@example.com and phone {i:03d}-{i:03d}-{i:04d}"
            for i in range(100)
        ]

        # Test detection performance
        start_time = time.time()
        for text in test_texts:
            detections = pii_filter.detect(text)
            assert len(detections) >= 2  # email and phone

        detection_time = time.time() - start_time
        per_text_time = detection_time / len(test_texts)

        assert per_text_time < 0.01, f"Per-text detection time: {per_text_time:.6f}s"

        # Test redaction performance
        start_time = time.time()
        for text in test_texts:
            redacted, found_pii, pii_types = pii_filter.redact(text)
            assert found_pii is True
            assert len(pii_types) >= 2

        redaction_time = time.time() - start_time
        per_text_time = redaction_time / len(test_texts)

        assert per_text_time < 0.01, f"Per-text redaction time: {per_text_time:.6f}s"

    def test_encryption_performance(self):
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
