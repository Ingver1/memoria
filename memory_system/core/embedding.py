# embedding.py — text-embedding backend for Unified Memory System
"""
High-level embedding service with:
- **Lazy model loading** & fallback to a lightweight default.
- **Smart cache** – TTL + LRU keeping memory footprint predictable.
- **Batch processor** – background thread pool that multiplexes single-text requests into vectorized *Sentence-Transformers* calls.
- **Graceful shutdown** – ensures no futures dangle after exit.

This implementation replaces earlier versions that had race conditions and redundant locks.
It is now async-friendly and uses English-only comments.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from memory_system.utils.dependencies import require_numpy, require_sentence_transformers

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from numpy import ndarray
    from sentence_transformers import SentenceTransformer
else:  # pragma: no cover - runtime helper
    SentenceTransformer = Any

from memory_system.settings import UnifiedSettings
from memory_system.utils.cache import SmartCache
from memory_system.utils.loop import get_loop
from memory_system.utils.metrics import (
    EMBEDDING_QUEUE_LENGTH,
    LAT_EMBEDDING_WAIT,
    MET_ERRORS_TOTAL,
)

__all__ = ["EmbeddingError", "EmbeddingJob", "EmbeddingService", "EnhancedEmbeddingService"]

log = logging.getLogger(__name__)


###############################################################################
# Exceptions & Data Containers
###############################################################################


class EmbeddingError(RuntimeError):
    """Raised when the embedding pipeline fails irrecoverably."""


@dataclass(slots=True, frozen=True)
class EmbeddingJob:
    """Internal container binding a text to its awaiting Future result."""

    text: str
    future: asyncio.Future[ndarray]
    enqueued_at: float = field(default_factory=time.monotonic)


###############################################################################
# Embedding Service Implementation
###############################################################################


class EmbeddingService:
    """
    Thread-safe, cache-aware embedding service.

    Public API consists of the `embed_text` method (async) and a few internal helpers.
    All heavy lifting happens in background worker threads to keep the asyncio event loop responsive.
    """

    def __init__(self, model_name: str, settings: UnifiedSettings | None = None) -> None:
        """Initialise the service and start the batching threads."""
        self.model_name = model_name
        self.settings = settings or UnifiedSettings.for_development()
        self._model_lock = threading.RLock()
        self._model: SentenceTransformer | None = None

        # Initialize cache for embeddings (size and TTL from settings)
        self.cache = SmartCache(
            max_size=self.settings.performance.cache_size,
            ttl=self.settings.performance.cache_ttl_seconds,
        )

        # Batch processing state
        self._queue: list[EmbeddingJob] = []
        self._queue_lock = threading.RLock()
        self._queue_condition = threading.Condition(self._queue_lock)
        self._shutdown = threading.Event()
        EMBEDDING_QUEUE_LENGTH.set(0)

        # Background batch processing threads
        self._batch_threads: list[threading.Thread] = []

        # Eager initialization: load model and start batch thread
        self._load_model(self.model_name)
        self._start_processor()  # start background batching threads
        log.info("Embedding service ready (model=%s)", self.model_name)

    # Context manager support
    async def __aenter__(self) -> EmbeddingService:
        """Enter the service context asynchronously."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any | None,
    ) -> None:
        """Ensure the service is closed when leaving an async context."""
        await self.close()

    def __enter__(self) -> EmbeddingService:
        """
        Synchronous wrapper for the async context manager.

        Avoids creating new event loops in restricted sandboxes by reusing the
        project loop helper when available.
        """
        loop = None
        try:
            loop = get_loop()
        except RuntimeError:
            pass
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.__aenter__(), loop)
            return fut.result()
        if loop is not None:
            return loop.run_until_complete(self.__aenter__())
        # No usable loop available
        raise RuntimeError("no usable event loop available")

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any | None
    ) -> None:
        """
        Synchronous wrapper for the async context manager.

        May raise :class:`RuntimeError` if an event loop is already running.
        """
        loop = None
        try:
            loop = get_loop()
        except RuntimeError:
            pass
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self.__aexit__(exc_type, exc, tb), loop)
            fut.result()
            return
        if loop is not None:
            loop.run_until_complete(self.__aexit__(exc_type, exc, tb))
            return
        asyncio.run(self.__aexit__(exc_type, exc, tb))

    # Model management

    def _load_model(self, name: str) -> None:
        """Lazy-load a SentenceTransformer model."""
        with self._model_lock:
            if self._model is not None:
                return
            try:
                log.info("Loading embedding model: %s", name)
                st = require_sentence_transformers()
                model = st(name)
                self._model = model
                self.model_name = name
                log.info(
                    "Model loaded: %s (dim=%d)",
                    name,
                    model.get_sentence_embedding_dimension(),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                log.warning("Primary model loading failed: %s", exc)
                fallback = "all-MiniLM-L6-v2"
                if name != fallback:
                    try:
                        log.info("Attempting fallback model: %s", fallback)
                        st = require_sentence_transformers()
                        model = st(fallback)
                        self._model = model
                        self.model_name = fallback
                        log.info("Fallback model loaded: %s", fallback)
                        return
                    except (OSError, RuntimeError, ValueError) as fexc:
                        log.error("Fallback model failed to load: %s", fexc)
                        raise EmbeddingError("Could not load any embedding model") from fexc
                raise EmbeddingError("Could not load embedding model") from exc

    def _start_processor(self) -> None:
        """Start the background batching threads (idempotent)."""
        if self._batch_threads and all(t.is_alive() for t in self._batch_threads):
            return
        self._batch_threads = []
        for i in range(self.settings.performance.max_workers):
            t = threading.Thread(target=self._batch_loop, name=f"emb-batcher-{i}", daemon=True)
            t.start()
            self._batch_threads.append(t)

    def _batch_loop(self) -> None:
        """Background thread loop that processes queued embedding jobs in batches."""
        np = require_numpy()
        batch_size = self.settings.model.batch_add_size
        log.debug("Batch processor loop started (batch_size=%d)", batch_size)
        while not self._shutdown.is_set():
            with self._queue_condition:
                if not self._queue:
                    # Wait for a short time for new jobs
                    self._queue_condition.wait(timeout=0.05)
                    continue
                # Slice a batch from the queue
                batch = self._queue[:batch_size]
                self._queue = self._queue[batch_size:]
                EMBEDDING_QUEUE_LENGTH.set(len(self._queue))
            # Process batch outside the lock
            try:
                now = time.monotonic()
                for job in batch:
                    LAT_EMBEDDING_WAIT.observe(now - job.enqueued_at)
                texts = [job.text for job in batch]
                embeddings = self._embed_direct(texts)  # synchronous call
                for job, embedding in zip(batch, embeddings, strict=True):
                    if not job.future.done():
                        loop = job.future.get_loop()
                        loop.call_soon_threadsafe(
                            job.future.set_result,
                            np.asarray(embedding),
                        )
            except Exception as exc:
                """Handle any embedding errors and propagate them to waiting jobs.

                The batch processor runs in a background thread and calls the
                synchronous ``_embed_direct`` method.  If that method raises an
                exception that we don't catch here, the exception would terminate
                the thread silently and any futures waiting for a result would
                hang.  This was most visible when a ``TimeoutError`` was raised
                during the tests, leaving the embedding thread alive and the
                pending futures unresolved.  To make the service robust we catch
                all ``Exception`` instances and convert them into
                ``EmbeddingError`` objects which are delivered to the awaiting
                futures.
                """
                MET_ERRORS_TOTAL.labels(type="embedding", component="batch_loop").inc()
                for job in batch:
                    if not job.future.done():
                        loop = job.future.get_loop()
                        loop.call_soon_threadsafe(
                            job.future.set_exception,
                            EmbeddingError(str(exc)),
                        )
        log.debug("Batch processor loop exited")

    # Public API

    async def embed_text(self, text: str | Sequence[str]) -> ndarray:
        """Return an embedding for the given text (string or sequence of strings)."""
        if isinstance(text, str):
            # Single text -> returns shape (1, dim) array
            return await self._embed_single(text)
        # Sequence of texts -> returns shape (n, dim) array
        return await self._embed_multi(list(text))

    # Internal async helpers

    async def _embed_single(self, text: str) -> ndarray:
        """Embed a single string into a 1 x dim embedding vector (as numpy array)."""
        require_numpy()
        if not text:
            raise ValueError("text must not be empty")
        # Attempt cache lookup first
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            # Return cached embedding, ensure shape (1, dim)
            return cast("ndarray", cached).reshape(1, -1)
        # Not in cache: enqueue for batch processing
        loop = get_loop()
        future: asyncio.Future[ndarray] = loop.create_future()
        job = EmbeddingJob(text=text, future=future, enqueued_at=time.monotonic())
        with self._queue_condition:
            if len(self._queue) >= self.settings.performance.queue_max_size:
                EMBEDDING_QUEUE_LENGTH.set(len(self._queue))
                MET_ERRORS_TOTAL.labels(type="embedding", component="queue").inc()
                raise EmbeddingError("Embedding queue is full")
            self._queue.append(job)
            EMBEDDING_QUEUE_LENGTH.set(len(self._queue))
            self._queue_condition.notify_all()
        try:
            embedding = await asyncio.wait_for(future, timeout=30.0)
        except TimeoutError:
            future.cancel()
            raise EmbeddingError("Embedding timed out") from None
        # Cache the new embedding result for future reuse
        self.cache.put(key, embedding)
        return embedding.reshape(1, -1)

    async def _embed_multi(self, texts: list[str]) -> ndarray:
        """Embed a list of texts into an array of embeddings."""
        np = require_numpy()
        if not texts:
            return cast("ndarray", np.empty((0, 0), dtype=np.float32))

        # Embed all texts in one go using the direct encoder in a worker thread
        embeddings = await asyncio.to_thread(self._embed_direct, texts)

        # Persist each individual embedding in the cache
        for text, embedding in zip(texts, embeddings, strict=True):
            key = self._cache_key(text)
            if self.cache.get(key) is None:
                self.cache.put(key, embedding)

        return embeddings

    def _embed_direct(self, texts: list[str]) -> ndarray:
        """Directly embed a batch of texts (runs in background thread)."""
        np = require_numpy()
        if self._model is None:
            raise EmbeddingError("Embedding model is not loaded")
        vecs = self._model.encode(texts)
        return cast("ndarray", np.asarray(vecs, dtype=np.float32))

    def _cache_key(self, text: str) -> str:
        """Compute a cache key for a given text input."""
        # Use SHA-256 for caching to reduce collision risk
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def close(self) -> None:
        """Gracefully shut down the embedding service, stopping the batch thread."""
        self._shutdown.set()
        with self._queue_condition:
            for job in self._queue:
                if not job.future.done():
                    loop = job.future.get_loop()
                    loop.call_soon_threadsafe(job.future.cancel)
            self._queue.clear()
            EMBEDDING_QUEUE_LENGTH.set(0)
            self._queue_condition.notify_all()

        for t in self._batch_threads:
            t.join()
        log.info("Embedding service closed.")

    # Convenience synchronous wrapper used in tests
    def shutdown(self) -> None:
        """Synchronously close the service for test helpers."""
        try:
            loop = get_loop()
        except RuntimeError:
            # No usable loop in this sandbox; perform best-effort shutdown
            self._shutdown.set()
            with self._queue_condition:
                self._queue.clear()
                EMBEDDING_QUEUE_LENGTH.set(0)
                self._queue_condition.notify_all()
            for t in self._batch_threads:
                t.join()
            return
        if loop.is_running():
            loop.create_task(self.close())
        else:
            loop.run_until_complete(self.close())

    def stats(self) -> dict[str, Any]:
        """Return basic runtime statistics for tests."""
        model = self._model
        return {
            "model": self.model_name,
            "dimension": model.get_sentence_embedding_dimension() if model else 0,
            "cache": self.cache.get_stats(),
            "queue_size": len(self._queue),
            "shutdown": self._shutdown.is_set(),
        }


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------

# Earlier versions exposed ``EnhancedEmbeddingService``.  Provide a simple alias
# so that external code and tests referencing the old name continue to work
# without modification.
EnhancedEmbeddingService = EmbeddingService
