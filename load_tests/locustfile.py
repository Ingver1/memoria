"""Load-test for FastAPI /search & /add endpoints using Locust."""

from typing import cast

import numpy as np
from locust import HttpUser, between, task

DIM = 384  # keep in sync with settings.model.vector_dim


def rand_vec() -> list[float]:
    """Return a random float32 vector as a JSON-serialisable list."""
    return cast("list[float]", np.random.rand(DIM).astype("float32").tolist())


class MemoryServiceUser(HttpUser):
    """Simulate interaction with the memory service."""

    wait_time = between(0.1, 1.0)

    @task(2)
    def add_memory(self) -> None:
        """Add a sample memory record via the API."""
        self.client.post(
            "/memory",
            json={"text": "locust-load", "embedding": rand_vec()},
            timeout=30,
        )

    @task(3)
    def search(self) -> None:
        """Search the API using a random vector."""
        self.client.post(
            "/search",
            json={"vector": rand_vec(), "k": 5},
            timeout=30,
        )
