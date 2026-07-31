"""Structured (JSON) logging using only the standard library.

We emit one JSON object per log record so that request-level observability
(retrieved chunk ids, scores, latency, token usage) is machine-parseable
without pulling in a logging framework.

Three observability building blocks live here:

* **Timestamp** — every record carries an ISO-8601 UTC ``timestamp`` derived
  from ``record.created`` so logs can be ordered/correlated after the fact.
* **Correlation id** — a request/chain GUID is carried on a
  ``contextvars.ContextVar`` and injected into every record by a
  ``logging.Filter``. This propagates through the call chain without threading
  an id through every function signature (the API middleware sets it per
  request). It is a join key, not PII / user identity.
* **Dedicated ``rag`` logger + stdout/stderr split** — we configure our own
  ``rag`` logger (not the root logger) so third-party logs (chromadb, openai,
  uvicorn, httpx) stay isolated from our structured app logs. Happy-path
  records (INFO and below) go to stdout; problems (WARNING and above) go to
  stderr, which must not be rate-limited/dropped.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Standard LogRecord attributes we don't want to duplicate inside "extra".
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}

# JSON keys the formatter owns. An ``extra`` field colliding with one of these
# is namespaced (``extra_<key>``) rather than dropped, so no information is lost.
_BASE_KEYS = frozenset({"timestamp", "level", "logger", "message", "correlation_id"})

# The dedicated application logger. All app modules use ``rag.*`` child loggers
# (e.g. ``rag.pipeline``) which propagate up to the handlers installed here.
APP_LOGGER_NAME = "rag"

# Carries the current request/chain correlation id. Empty string means "no
# correlation context" (e.g. a log emitted outside any request).
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def set_correlation_id(value: str) -> contextvars.Token:
    """Bind ``value`` as the current correlation id; returns a reset token."""

    return correlation_id_var.set(value)


def reset_correlation_id(token: contextvars.Token) -> None:
    """Restore the correlation id to its previous value (pair with ``set``)."""

    correlation_id_var.reset(token)


def get_correlation_id() -> str:
    """Return the current correlation id (empty string if none is set)."""

    return correlation_id_var.get()


class CorrelationIdFilter(logging.Filter):
    """Stamp every record with the current correlation id from the contextvar."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


class _MaxLevelFilter(logging.Filter):
    """Allow only records at or below ``max_level`` (for the stdout handler)."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self._max_level


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Anything passed via ``logger.info(msg, extra={...})`` is merged into the
    top level of the JSON object, so structured fields are first-class. An
    ``extra`` key that collides with a base key is namespaced (``extra_<key>``)
    instead of dropped.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation_id = getattr(record, "correlation_id", "")
        if correlation_id:
            payload["correlation_id"] = correlation_id
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key == "correlation_id":
                continue
            # Namespace a collision with a base key rather than discard it.
            target = key if key not in _BASE_KEYS else f"extra_{key}"
            payload[target] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the dedicated ``rag`` logger (idempotent).

    Installs a stdout handler (INFO and below) and a stderr handler
    (WARNING and above), both emitting JSON with the correlation id injected.
    The root logger is left untouched so third-party library logs stay
    isolated from our structured app logs.
    """

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level.upper())
    # Don't hand our records to the root logger — keeps third-party handlers
    # (and their noise/cost) out of our structured stream.
    app_logger.propagate = False

    # Replace existing handlers so repeated calls don't duplicate output.
    for handler in list(app_logger.handlers):
        app_logger.removeHandler(handler)

    formatter = JsonFormatter()
    correlation_filter = CorrelationIdFilter()

    # Happy path (INFO and below) -> stdout. This stream can be rate-limited.
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(correlation_filter)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))

    # Problems (WARNING and above) -> stderr. This stream must not be dropped.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(correlation_filter)
    stderr_handler.setLevel(logging.WARNING)

    app_logger.addHandler(stdout_handler)
    app_logger.addHandler(stderr_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
