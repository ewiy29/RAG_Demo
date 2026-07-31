"""Tests for WS8 observability: JSON timestamp, correlation id (contextvar +
filter + HTTP propagation), dedicated ``rag`` logger with stdout/stderr split,
extra-collision namespacing, and pipeline failure logging.

The formatter/filter/handler wiring is exercised directly (constructing records)
so the assertions don't depend on global logging state; the correlation-id HTTP
propagation is exercised through the ASGI app like the other API tests.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime

import httpx
import pytest
from httpx import ASGITransport

from doubles import build_fake_pipeline
from rag.api import CORRELATION_ID_HEADER, create_app
from rag.config import Settings
from rag.logging_utils import (
    APP_LOGGER_NAME,
    CorrelationIdFilter,
    JsonFormatter,
    configure_logging,
    reset_correlation_id,
    set_correlation_id,
)


def _make_record(level: int = logging.INFO, msg: str = "hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="rag.test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    # Simulate ``logger.x(msg, extra={...})`` fields landing on the record.
    for key, value in extra.items():
        record.__dict__[key] = value
    return record


# --- formatter -------------------------------------------------------------


def test_formatter_includes_iso_timestamp_level_logger_message():
    payload = json.loads(JsonFormatter().format(_make_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag.test"
    assert payload["message"] == "hello"
    # ISO-8601, timezone-aware (UTC), and parseable back to a datetime.
    parsed = datetime.fromisoformat(payload["timestamp"])
    assert parsed.tzinfo is not None


def test_formatter_merges_extra_fields_first_class():
    payload = json.loads(JsonFormatter().format(_make_record(user_id="u1", grounded=True)))

    assert payload["user_id"] == "u1"
    assert payload["grounded"] is True


def test_extra_collision_is_namespaced_not_dropped():
    # ``logger`` / ``timestamp`` are base JSON keys but are not reserved
    # LogRecord attributes, so an extra can collide with them; the collision
    # must be kept, namespaced, rather than dropped.
    record = _make_record(logger="shadow-logger")
    record.__dict__["timestamp"] = "shadow-timestamp"
    payload = json.loads(JsonFormatter().format(record))

    assert payload["logger"] == "rag.test"  # base key wins its slot
    assert payload["timestamp"] != "shadow-timestamp"
    assert payload["extra_logger"] == "shadow-logger"  # collision preserved
    assert payload["extra_timestamp"] == "shadow-timestamp"


# --- correlation id --------------------------------------------------------


def test_correlation_filter_stamps_current_contextvar():
    token = set_correlation_id("corr-123")
    try:
        record = _make_record()
        assert CorrelationIdFilter().filter(record) is True
        payload = json.loads(JsonFormatter().format(record))
        assert payload["correlation_id"] == "corr-123"
    finally:
        reset_correlation_id(token)


def test_correlation_id_omitted_when_unset():
    record = _make_record()
    CorrelationIdFilter().filter(record)  # default is empty string
    payload = json.loads(JsonFormatter().format(record))
    assert "correlation_id" not in payload


# --- dedicated logger + stdout/stderr split --------------------------------


def test_configure_logging_uses_dedicated_logger_with_split_handlers():
    configure_logging("INFO")
    app_logger = logging.getLogger(APP_LOGGER_NAME)

    # Own logger, isolated from root (third-party logs stay out of our stream).
    assert app_logger.propagate is False
    assert len(app_logger.handlers) == 2

    stdout_handler = next(h for h in app_logger.handlers if h.stream is sys.stdout)
    stderr_handler = next(h for h in app_logger.handlers if h.stream is sys.stderr)

    # Both carry the correlation-id filter.
    for handler in (stdout_handler, stderr_handler):
        assert any(isinstance(f, CorrelationIdFilter) for f in handler.filters)

    # Happy path (<= INFO) -> stdout; a WARNING is filtered off the stdout handler.
    assert stdout_handler.filter(_make_record(level=logging.INFO))  # truthy: passes
    assert not stdout_handler.filter(_make_record(level=logging.WARNING))

    # Problems (>= WARNING) -> stderr; an INFO record is below its level.
    assert stderr_handler.level == logging.WARNING


# --- correlation id over HTTP ----------------------------------------------


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_response_carries_a_minted_correlation_id():
    pipe = build_fake_pipeline(Settings())
    async with _client(create_app(pipeline=pipe)) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    minted = resp.headers.get(CORRELATION_ID_HEADER)
    assert minted  # a fresh id is minted when none is supplied
    assert len(minted) >= 8


async def test_inbound_correlation_id_is_echoed_back():
    pipe = build_fake_pipeline(Settings())
    async with _client(create_app(pipeline=pipe)) as client:
        resp = await client.get(
            "/health", headers={CORRELATION_ID_HEADER: "trace-abc"}
        )

    # An upstream id continues through unchanged so logs join across services.
    assert resp.headers.get(CORRELATION_ID_HEADER) == "trace-abc"


# --- pipeline failure logging ----------------------------------------------


class _ListHandler(logging.Handler):
    """Capture emitted records directly off a logger (propagation-independent)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _BoomEmbedder:
    async def embed(self, texts):
        raise RuntimeError("embed exploded")


async def test_ask_failure_is_logged_and_reraised():
    pipe = build_fake_pipeline(Settings())
    # Force the retrieval step to blow up.
    pipe.embedder = _BoomEmbedder()

    pipeline_logger = logging.getLogger("rag.pipeline")
    handler = _ListHandler()
    previous_level = pipeline_logger.level
    pipeline_logger.addHandler(handler)
    pipeline_logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(RuntimeError, match="embed exploded"):
            await pipe.ask("anything?", user_id="u1")
    finally:
        pipeline_logger.removeHandler(handler)
        pipeline_logger.setLevel(previous_level)

    failed = next(r for r in handler.records if r.getMessage() == "ask_failed")
    assert failed.levelno == logging.ERROR
    assert failed.exc_info is not None  # stack captured for diagnosis
    assert failed.__dict__["user_id"] == "u1"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
