"""Structured (JSON) logging using only the standard library.

We emit one JSON object per log record so that request-level observability
(retrieved chunk ids, scores, latency, token usage) is machine-parseable
without pulling in a logging framework.
"""

from __future__ import annotations

import json
import logging
from typing import Any

# Standard LogRecord attributes we don't want to duplicate inside "extra".
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON.

    Anything passed via ``logger.info(msg, extra={...})`` is merged into the
    top level of the JSON object, so structured fields are first-class.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""

    root = logging.getLogger()
    root.setLevel(level.upper())

    # Replace existing handlers so repeated calls don't duplicate output.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
