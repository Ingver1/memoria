# embedding.py — text-embedding backend for Unified Memory System
"""High-level embedding service with:
- **Lazy model loading** & fallback to a lightweight default.
- **Smart cache** – TTL + LRU keeping memory footprint predictable.
- **Batch processor** – background thread that multiplexes single-text requests into vectorized *Sentence-Transformers* calls.
- **Graceful shutdown** – ensures no futures dangle after exit.

This implementation replaces earlier versions that had race conditions and redundant locks.
It is now async-friendly and uses English-only comments.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

try:  # optional numpy
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:  # optional sentence-transformers
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as _np

from memory_system.config.settings import UnifiedSettings
from memory_system.utils.cache import SmartCache
from memory_system.utils.loop import get_or_create_loop
from memory_system.utils.metrics import EMBEDDING_QUEUE_LENGTH, MET_ERRORS_TOTAL

__all__ = ["EmbeddingError", "EmbeddingJob", "EnhancedEmbeddingService"]

log = logging.getLogger(__name__)


def _require_numpy() -> Any:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for embedding services. Install ai-memory[embedding]."
        )
    return np


def _require_sentence_transformers() -> Any:
    if SentenceTransformer is None:
        raise ModuleNotFoundError(
            "sentence-transformers is required for embedding services."
        )
    return SentenceTransformer

###############################################################################
# Exceptions & Data Containers
###############################################################################


class EmbeddingError(RuntimeError):
    """Raised when the embedding pipeline fails irrecoverably."""


@dataclass(slots=True, frozen=True)
class EmbeddingJob:
    """Internal container binding a text to its awaiting Future result."""

    text: str
    future: asyncio.Future[np.ndarray]


###############################################################################
# Embedding Service Implementation
###############################################################################


class EmbeddingService:
    """Thread-safe, cache-aware embedding service.

    Public API consists of the `embed_text` method (async) and a few internal helpers.
    All heavy lifting happens in a single background thread to keep the asyncio event loop responsive.
    """

    def __init__(self, model_name: str, settings: UnifiedSettings | None = None) -> None:
        """Initialise the service and start the batching thread."""
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

        # Background batch processing thread handle
        self._batch_thread: threading.Thread | None = None

        # Eager initialization: load model and start batch thread
        self._load_model()  # load model (or fallback)
        self._start_processor()  # start background batching thread
        log.info("Embedding service ready (model=%s)", self.model_name)

    # Context manager support
    def __enter__(self) -> "EmbeddingService":
        """Enter the service context."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any | None) -> None:
        """Ensure the service is closed when leaving a context."""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # In case an event loop is already running, schedule close
            loop = get_or_create_loop()
            loop.run_until_complete(self.close())

    # Model management

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model with a fallback on failure."""
        with self._model_lock:
            if self._model is not None:
                return
            try:
                log.info("Loading embedding model: %s", self.model_name)
                st = _require_sentence_transformers()
                self._model = st(self.model_name)
                log.info(
                    "Model loaded: %s (dim=%d)",
                    self.model_name,
                    self._model.get_sentence_embedding_dimension(),
                )
            except Exception as exc:
                log.warning("Primary model loading failed: %s", exc)
                fallback = "all-MiniLM-L6-v2"
                if self.model_name != fallback:
                    try:
                        log.info("Attempting fallback model: %s", fallback)
                        st = _require_sentence_transformers()
                        self._model = st(fallback)
                        self.model_name = fallback
                        log.info("Fallback model loaded: %s", fallback)
                    except Exception as fexc:
                        log.error("Fallback model failed to load: %s", fexc)
                        raise EmbeddingError("Could not load any embedding model") from fexc
                else:
                    raise EmbeddingError("Could not load embedding model") from exc

    def _start_processor(self) -> None:
        """Start the background batching thread (idempotent)."""
        if self._batch_thread and self._batch_thread.is_alive():
            return
        self._batch_thread = threading.Thread(target=self._batch_loop, name="emb-batcher", daemon=True)
        self._batch_thread.start()

    def _batch_loop(self) -> None:
        """Background thread loop that processes queued embedding jobs in batches."""
        np = _require_numpy()
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
                texts = [job.text for job in batch]
                embeddings = self._embed_direct(texts)  # synchronous call
                for job, embedding in zip(batch, embeddings, strict=False):
                    if not job.future.done():
                        loop = job.future.get_loop()
                        loop.call_soon_threadsafe(
                            job.future.set_result,
                            np.asarray(embedding),
                        )
            except Exception as exc:
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

    async def embed_text(self, text: str | Sequence[str]) -> "_np.ndarray":
        """Return an embedding for the given text (string or sequence of strings)."""
        if isinstance(text, str):
            # Single text -> returns shape (1, dim) array
            return await self._embed_single(text)
        else:
            # Sequence of texts -> returns shape (n, dim) array
            return await self._embed_multi(list(text))

    # Internal async helpers

    async def _embed_single(self, text: str) -> "_np.ndarray":
        """Embed a single string into a 1 x dim embedding vector (as numpy array)."""
        np = _require_numpy()
        if not text:
            raise ValueError("text must not be empty")
        # Attempt cache lookup first
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            # Return cached embedding, ensure shape (1, dim)
            return cast(np.ndarray, cached).reshape(1, -1)
        # Not in cache: enqueue for batch processing
        await asyncio.sleep(0.001)
        loop = get_or_create_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        job = EmbeddingJob(text=text, future=future)
        with self._queue_condition:
            if len(self._queue) >= self.settings.performance.queue_max_size:
                EMBEDDING_QUEUE_LENGTH.set(len(self._queue))
                raise EmbeddingError("Embedding queue is full")
            self._queue.append(job)
            EMBEDDING_QUEUE_LENGTH.set(len(self._queue))
            self._queue_condition.notify()
        try:
            embedding = await asyncio.wait_for(future, timeout=30.0)
        except TimeoutError:
            future.cancel()
            raise EmbeddingError("Embedding timed out") from None
        # Cache the new embedding result for future reuse
        self.cache.put(key, embedding)
        return embedding.reshape(1, -1)

    async def _embed_multi(self, texts: list[str]) -> "_np.ndarray":
        """Embed a list of texts into an array of embeddings."""
        np = _require_numpy()
        # For multiple texts, we can process them directly (and possibly cache each)
        embeddings = []
        for text in texts:
            embedding = await self._embed_single(text)
            embeddings.append(embedding)  # each embedding is 1 x dim
        # Concatenate results into one array
        return cast(np.ndarray, np.vstack(embeddings))

    def _embed_direct(self, texts: list[str]) -> "_np.ndarray":
        """Directly embed a batch of texts (runs in background thread)."""
        np = _require_numpy()
        if self._model is None:
            raise EmbeddingError("Embedding model is not loaded")
        embedding = self._model.encode(texts)
        return cast(np.ndarray, np.asarray(embedding, dtype=np.float32))

    def _cache_key(self, text: str) -> str:
        """Compute a cache key for a given text input."""
        # Use a hash for caching (e.g., SHA1 for speed, here MD5 via hashlib for simplicity)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    async def close(self) -> None:
        """Gracefully shut down the embedding service, stopping the batch thread."""
        self._shutdown.set()
        # Wait for batch thread to finish if running
        if self._batch_thread:
            self._batch_thread.join(timeout=1.0)
        log.info("Embedding service closed.")

    # Convenience synchronous wrapper used in tests
    def shutdown(self) -> None:
        """Synchronously close the service for test helpers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            loop.create_task(self.close())
        else:
            asyncio.run(self.close())

    def stats(self) -> dict[str, Any]:
        """Return basic runtime statistics for tests."""
        return {
            "model": self.model_name,
            "dimension": self._model.get_sentence_embedding_dimension() if self._model else 0,
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
