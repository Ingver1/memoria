"""Logging helpers for the Unified Memory System."""

from __future__ import annotations

import logging

from memory_system.context import REQUEST_ID


def _add_request_id(record: logging.LogRecord) -> logging.LogRecord:
    record.request_id = REQUEST_ID.get("-")
    return record


class RequestIdFilter(logging.Filter):
    """Inject ``request_id`` from :mod:`contextvars` into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        """Add the ``request_id`` attribute to *record* from context."""
        _add_request_id(record)
        return True


def setup_request_id_logging() -> None:
    """Ensure all log records include ``request_id`` attribute."""
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        return _add_request_id(old_factory(*args, **kwargs))

    logging.setLogRecordFactory(record_factory)


__all__ = ["RequestIdFilter", "setup_request_id_logging"]
