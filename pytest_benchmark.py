"""Minimal stub for pytest-benchmark to satisfy tests when the real package is absent."""

# The real plugin exposes a 'benchmark' fixture. We just define a placeholder
# so that `pytest.importorskip("pytest_benchmark")` succeeds.

__all__: list[str] = []
