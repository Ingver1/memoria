"""Test suite for Unified Memory System.

This package contains comprehensive tests for all components of the
Unified Memory System including:
- Unit tests for individual modules
- Integration tests for API endpoints
- End-to-end tests for complete workflows
- Performance tests
- Security tests
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest_plugins = ("pytest_asyncio",)

# Test configuration
TEST_DB_PATH = Path(tempfile.mkdtemp()) / "test_memory.db"
TEST_VECTOR_PATH = Path(tempfile.mkdtemp()) / "test_vectors.bin"
TEST_CACHE_PATH = Path(tempfile.mkdtemp()) / "test_cache"

# Ensure test directories exist
TEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TEST_VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
TEST_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# Test environment variables
os.environ.update(
    {
        "UMS_DB_PATH": str(TEST_DB_PATH),
        "UMS_VECTOR_PATH": str(TEST_VECTOR_PATH),
        "UMS_CACHE_PATH": str(TEST_CACHE_PATH),
        "UMS_LOG_LEVEL": "DEBUG",
        "UMS_ENVIRONMENT": "testing",
    }
)

__all__ = [
    "TEST_DB_PATH",
    "TEST_VECTOR_PATH",
    "TEST_CACHE_PATH",
]
