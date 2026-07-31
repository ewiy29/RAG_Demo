"""Tests for the API's structured error envelope + catch-all handler.

Domain errors reach the client as ``{"error": {domain, code, context}}`` with
the code's HTTP status; any unexpected exception becomes a generic INTERNAL 500
with no leaked prose or traceback.
"""

from __future__ import annotations

import httpx
from httpx import ASGITransport

from rag.api import create_app
from rag.config import Settings
from rag.errors import ProviderError, ProviderErrorCode, StoreError, StoreErrorCode


class _StubPipeline:
    """Duck-typed pipeline whose operations raise a preset exception."""

    def __init__(self, settings: Settings, exc: Exception) -> None:
        self.settings = settings
        self._exc = exc

    async def ask(self, query: str, *, user_id: str, conversation_id=None):
        raise self._exc

    async def ingest_uploads(self, uploads, *, user_id: str):
        raise self._exc


def _client(exc: Exception, **settings_overrides) -> httpx.AsyncClient:
    settings = Settings(**settings_overrides)
    app = create_app(pipeline=_StubPipeline(settings, exc))
    # raise_app_exceptions=False so the catch-all handler's response is observed
    # as an HTTP 500 rather than re-raised into the test.
    return httpx.AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


async def test_provider_error_becomes_structured_envelope():
    async with _client(
        ProviderError(ProviderErrorCode.RATE_LIMITED, context={"retry_after": 5})
    ) as client:
        resp = await client.post("/ask", json={"query": "anything?"})
    assert resp.status_code == 429
    assert resp.json() == {
        "error": {
            "domain": "provider",
            "code": "RATE_LIMITED",
            "context": {"retry_after": 5},
        }
    }


async def test_store_error_on_ingest_becomes_structured_envelope():
    async with _client(
        StoreError(StoreErrorCode.UNAVAILABLE),
    ) as client:
        resp = await client.post(
            "/ingest", files={"files": ("notes.md", b"some content", "text/markdown")}
        )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["domain"] == "store"
    assert body["error"]["code"] == "UNAVAILABLE"


async def test_unexpected_exception_becomes_generic_internal_500():
    async with _client(RuntimeError("boom: secret internal detail")) as client:
        resp = await client.post("/ask", json={"query": "anything?"})
    assert resp.status_code == 500
    assert resp.json() == {
        "error": {"domain": "internal", "code": "INTERNAL", "context": {}}
    }
    # No prose/traceback leaks into the response body.
    assert "boom" not in resp.text
    assert "secret internal detail" not in resp.text
