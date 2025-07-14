"""Lightweight psutil stub used in tests without the real dependency."""

from __future__ import annotations

from dataclasses import dataclass


class Process:
    """Very small subset of :mod:`psutil.Process`."""

    def __init__(self, pid: int | None = None) -> None:  # pragma: no cover - stub
        self.pid = pid

    @dataclass(slots=True)
    class MemoryInfo:
        rss: int = 0

    def memory_info(self) -> "Process.MemoryInfo":
        return self.MemoryInfo(0)


def cpu_percent(interval: float | None = None) -> float:  # pragma: no cover - stub
    """Return fake CPU usage percentage."""
    return 0.0


@dataclass(slots=True)
class _VirtualMemory:
    percent: float = 0.0


def virtual_memory() -> _VirtualMemory:  # pragma: no cover - stub
    """Return fake virtual memory information."""
    return _VirtualMemory(0.0)


__all__ = ["Process", "cpu_percent", "virtual_memory"]
