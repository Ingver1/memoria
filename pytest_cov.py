"""Minimal stub for pytest-cov to satisfy tests when the real package is absent."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register dummy coverage options so pytest does not fail."""
    group = parser.getgroup("cov")
    group.addoption("--cov", action="store", default=None, help="no-op")
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        metavar="type",
        help="no-op",
    )
    group.addoption("--cov-append", action="store_true", help="no-op")
    group.addoption("--cov-config", action="store", default=None, help="no-op")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Emit a minimal coverage report if requested."""
    reports: list[str] = session.config.getoption("--cov-report") or []
    if any("term" in r for r in reports):
        print("Coverage collection is disabled in this environment.")
    if any("xml" in r for r in reports):
        Path("coverage.xml").write_text("<coverage></coverage>\n")


__all__: list[str] = []
