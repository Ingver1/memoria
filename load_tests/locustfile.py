"""
Locust load-test for FastAPI /search & /add endpoints.
Run with: locust -f load_tests/locustfile.py --host http://localhost:8000
"""
import random

from locust import HttpUser, between, task

import numpy as np

DIM = 384  # keep in sync with settings.model.vector_dim


def rand_vec() -> list[float]:
    """Return random float32 vector as JSON-serialisable list."""
    return np.random.rand(DIM).astype("float32").tolist()


    wait_time = between(0.1, 1.0)

    @task(2)
    def add_memory(self) -> None:
        self.client.post(
            "/memory",
            json={"text": "locust-load", "embedding": rand_vec()},
            timeout=30,
        )

    @task(3)
    def search(self) -> None:
        self.client.post(
            "/search",
            json={"vector": rand_vec(), "k": 5},
            timeout=30,
        )

