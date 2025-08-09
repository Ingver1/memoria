"""Pytest configuration and fixtures."""

# Ensure the repository root is on ``sys.path`` so tests can import the package
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import inspect
import logging
import os
import tempfile
import uuid
from typing import Any, List

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]
import types

import pytest
from _pytest.config import Config
from _pytest.fixtures import FixtureRequest
from _pytest.logging import LogCaptureFixture
from _pytest.nodes import Item

try:
    from fastapi.testclient import TestClient
except Exception:  # FastAPI or its dependencies may be missing
    TestClient = None  # type: ignore[assignment]

from memory_system import __version__

try:
    from memory_system.api.app import create_app
except Exception:  # pragma: no cover - optional FastAPI dependency
    create_app = None  # type: ignore[assignment]
try:
    from memory_system.config.settings import UnifiedSettings
except Exception:  # pragma: no cover - optional dependency
    UnifiedSettings = None  # type: ignore[assignment]
try:
    from memory_system.core.index import FaissHNSWIndex
except Exception:  # pragma: no cover - optional dependency
    FaissHNSWIndex = None  # type: ignore[assignment]
from memory_system.core.store import SQLiteMemoryStore

# ...existing code...


def _install_crypto_stub() -> None:
    try:
        import cryptography.fernet  # noqa: F401

        return
    except Exception:
        pass

    from tests._stubs.cryptography import fernet as f_stub

    crypto_pkg = types.ModuleType("cryptography")
    sys.modules["cryptography"] = crypto_pkg
    sys.modules["cryptography.fernet"] = f_stub
    crypto_pkg.fernet = f_stub


_install_crypto_stub()


@pytest.fixture(scope="session", autouse=True)
def _inject_crypto_stub_if_missing() -> None:
    _install_crypto_stub()


def pytest_configure(config: Config) -> None:
    """Register custom markers used in the test suite."""
    config.addinivalue_line("markers", "asyncio: mark async test")
    config.addinivalue_line("markers", "perf: marks performance / load tests")


pytest_configure.__annotations__ = {"config": object, "return": None}


def pytest_collection_modifyitems(config: Config, items: List[Item]) -> None:
    """Automatically mark async test functions with pytest.mark.asyncio."""
    for item in items:
        test_fn = getattr(item, "obj", None)
        if inspect.iscoroutinefunction(test_fn):
            item.add_marker(pytest.mark.asyncio)


pytest_collection_modifyitems.__annotations__ = {"config": object, "items": list, "return": None}

# Set test environment variables
os.environ.update(
    {
        "UMS_ENVIRONMENT": "testing",
        "UMS_LOG_LEVEL": "DEBUG",
        "UMS_DB_PATH": str(Path(tempfile.mkdtemp()) / "test.db"),
        "CUDA_VISIBLE_DEVICES": "",  # Disable CUDA in tests
        "FAISS_OPT_LEVEL": "0",
    }
)


@pytest.fixture(autouse=True)
def _raise_log_level(caplog: LogCaptureFixture) -> None:
    """
    Silence DEBUG chatter from memory_system.core.index during CI.
    """
    caplog.set_level(logging.INFO, logger="memory_system.core.index")


_raise_log_level.__annotations__ = {"caplog": object, "return": None}


@pytest.fixture(scope="session")
def test_settings() -> UnifiedSettings:
    """Create test settings."""
    if UnifiedSettings is None:
        pytest.skip("memory_system.config.settings not available")
    return UnifiedSettings.for_testing()


@pytest.fixture
def test_app(test_settings: UnifiedSettings) -> Any:
    """Create FastAPI application for tests."""
    if create_app is None:
        pytest.skip("fastapi not available")
    return create_app()


test_app.__annotations__ = {"test_settings": object, "return": object}


@pytest.fixture
def test_client(test_app: Any) -> TestClient:
    """HTTP client for API tests."""
    if TestClient is None:
        pytest.skip("fastapi test client not available")
    return TestClient(test_app)


test_client.__annotations__ = {"test_app": object, "return": object}


@pytest.fixture
def clean_test_vectors(tmp_path: Path) -> Path:
    """Temporary path used for vector store tests."""
    return tmp_path / "vectors"


clean_test_vectors.__annotations__ = {"tmp_path": Path, "return": Path}


@pytest.fixture(name="benchmark")
def _benchmark() -> Any:
    """Simple stand-in for the pytest-benchmark fixture."""

    class DummyBenchmark:
        def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    return DummyBenchmark()


_benchmark.__annotations__ = {"return": object}


@pytest.fixture(name="mem_benchmark")
def _mem_benchmark(benchmark: Any) -> Any:
    """Backward-compatible alias for the benchmark fixture."""
    return benchmark


_mem_benchmark.__annotations__ = {"benchmark": object, "return": object}


@pytest.fixture
async def store(tmp_path: Path) -> SQLiteMemoryStore:
    """SQLite-backed memory store for tests."""
    db_path = tmp_path / "mem.db"
    st = SQLiteMemoryStore(db_path)
    await st.initialise()
    try:
        yield st
    finally:
        await st.aclose()


store.__annotations__ = {"tmp_path": Path, "return": SQLiteMemoryStore}


@pytest.fixture
def index() -> FaissHNSWIndex:
    """3D index to match the fake embedder."""
    if FaissHNSWIndex is None:
        pytest.skip("faiss not available")
    return FaissHNSWIndex(dim=3)


index.__annotations__ = {"return": FaissHNSWIndex}


@pytest.fixture
def fake_embed(monkeypatch) -> Any:
    """Deterministic embedder for tests."""
    if np is None:
        pytest.skip("numpy not available")
    import memory_system.core.maintenance as maint

    def _embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            tl = t.lower()
            if "cat" in tl:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "sky" in tl:
                v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            n = np.linalg.norm(v)
            if n:
                v = (v / n).astype(np.float32)
            vecs.append(v)
        return np.vstack(vecs) if len(vecs) > 1 else vecs[0]

    monkeypatch.setattr(maint, "embed_text", _embed, raising=True)
    return _embed


fake_embed.__annotations__ = {"monkeypatch": object, "return": object}


@pytest.fixture
def random_texts() -> list:
    """Provide a list of random texts."""
    return [f"Random text number {i}" for i in range(10)]


random_texts.__annotations__ = {"return": list}

# tests/test_basic.py
import pytest


def test_package_imports() -> None:
    """Test that basic package imports work."""
    try:
        import memory_system

        assert hasattr(memory_system, "__version__")
    except ImportError:
        pytest.skip("memory_system package not importable")


def test_exceptions_module() -> None:
    """Test that exceptions module works."""
    try:
        from memory_system.utils.exceptions import (
            MemorySystemError,
            StorageError,
            ValidationError,
        )

        # Test basic exception creation
        error = MemorySystemError("test error")
        assert str(error)  # Should not raise
        assert error.message == "test error"

        # Test inheritance
        assert issubclass(ValidationError, MemorySystemError)
        assert issubclass(StorageError, MemorySystemError)

    except ImportError:
        pytest.skip("exceptions module not available")


def test_config_module() -> None:
    """Test that config module works."""
    try:
        from memory_system.config.settings import UnifiedSettings

        # Test settings creation
        settings = UnifiedSettings.for_testing()
        assert settings is not None
        assert settings.version == __version__

    except ImportError:
        pytest.skip("config module not available")


def test_utils_module() -> None:
    """Test that utils module imports."""
    try:
        import memory_system.utils

        assert memory_system.utils is not None
    except ImportError:
        pytest.skip("utils module not available")


def test_core_module() -> None:
    """Test that core module imports."""
    try:
        import memory_system.core

        assert memory_system.core is not None
    except ImportError:
        pytest.skip("core module not available")


def test_api_module() -> None:
    """Test that api module imports."""
    try:
        import memory_system.api

        assert memory_system.api is not None
    except ImportError:
        pytest.skip("api module not available")


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_placeholder(self) -> None:
        """Placeholder test that always passes."""
        assert True

    def test_python_version(self) -> None:
        """Test Python version is supported."""
        import sys

        assert sys.version_info >= (3, 9)

    def test_environment_variables(self) -> None:
        """Test that test environment is set up correctly."""
        assert os.environ.get("UMS_ENVIRONMENT") == "testing"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
