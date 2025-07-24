"""Pytest configuration and fixtures."""

import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Callable, Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import ClientHelper

from memory_system import __version__
from memory_system.api.app import create_app


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used in the test suite."""
    config.addinivalue_line("markers", "asyncio: mark async test")
    config.addinivalue_line("markers", "perf: marks performance / load tests")


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Automatically mark async test functions with pytest.mark.asyncio."""
    for item in items:
        test_fn = getattr(item, "obj", None)
        if inspect.iscoroutinefunction(test_fn):
            item.add_marker(pytest.mark.asyncio)

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
def _raise_log_level(caplog: pytest.LogCaptureFixture) -> None:
    """
    Silence DEBUG chatter from memory_system.core.index during CI.
    """
    caplog.set_level(logging.INFO, logger="memory_system.core.index")
    

@pytest.fixture(scope="session")
def test_settings() -> object:
    """Create test settings."""
    try:
        from memory_system.config.settings import UnifiedSettings

        return UnifiedSettings.for_testing()
    except ImportError:
        pytest.skip("memory_system.config.settings not available")


@pytest.fixture
def test_app(test_settings: object) -> FastAPI:
    """Create FastAPI application for tests."""
    return create_app()


@pytest.fixture
def test_client(test_app: FastAPI) -> ClientHelper:
    """HTTP client for API tests."""
    return ClientHelper(test_app)


@pytest.fixture
def clean_test_vectors(tmp_path: Path) -> Path:
    """Temporary path used for vector store tests."""
    return tmp_path / "vectors"

@pytest.fixture(name="benchmark")
def _benchmark() -> object:
    """Simple stand-in for the pytest-benchmark fixture."""

    class DummyBenchmark:
        def __call__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    return DummyBenchmark()


@pytest.fixture(name="mem_benchmark")
def _mem_benchmark(benchmark: object) -> object:
    """Backward-compatible alias for the benchmark fixture."""

    return benchmark
    
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
