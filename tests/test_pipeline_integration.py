"""End-to-end integration test for the full ingest -> ask pipeline.

Runs entirely offline against the fake provider and an in-memory store, over a
tiny fixture corpus. Asserts both the grounded-with-citation path and the
out-of-scope refusal path, at the pipeline level and through the HTTP API.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from doubles import build_fake_pipeline
from rag.api import create_app
from rag.config import Settings
from rag.generate import REFUSAL_TEXT

FIXTURES = Path(__file__).parent / "fixtures"

USER = "integration-user"


async def _pipeline():
    settings = Settings(top_k=4, min_score=0.2, chunk_size=400, chunk_overlap=50)
    pipe = build_fake_pipeline(settings)
    result = await pipe.ingest([str(FIXTURES)], user_id=USER)
    assert result.documents >= 2
    assert result.chunks >= 2
    return pipe


def _async_client(app) -> httpx.AsyncClient:
    # Drive the ASGI app directly (no live server) over the true async path.
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_in_scope_question_is_grounded_and_cites_source():
    pipe = await _pipeline()
    resp = await pipe.ask("What is water composed of?", user_id=USER)

    assert resp.grounded is True
    assert resp.answer != REFUSAL_TEXT
    assert resp.citations, "a grounded answer must carry at least one citation"
    # The citation must point back to the water fixture.
    assert any("water.md" in c.source for c in resp.citations)
    # The retrieved context that won should be the water document.
    assert "water.md" in resp.retrieved[0].source


async def test_out_of_scope_question_triggers_refusal():
    pipe = await _pipeline()
    resp = await pipe.ask("Who won the 1998 football world cup final?", user_id=USER)

    assert resp.grounded is False
    assert resp.answer == REFUSAL_TEXT
    assert resp.citations == []
    assert resp.retrieved == []  # nothing cleared the threshold


async def test_api_ingest_then_ask_grounded_and_refusal():
    settings = Settings(top_k=4, min_score=0.2)
    pipe = build_fake_pipeline(settings)

    # A stable tenant id shared by the ingest and both asks so retrieval is
    # scoped to the just-uploaded corpus.
    headers = {"X-User-Id": USER}

    async with _async_client(create_app(pipeline=pipe)) as client:
        # Ingest the water fixture via multipart upload.
        with open(FIXTURES / "water.md", "rb") as fh:
            r = await client.post(
                "/ingest",
                files={"files": ("water.md", fh.read(), "text/markdown")},
                headers=headers,
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1
        assert r.json()["user_id"] == USER
        assert r.headers["X-User-Id"] == USER

        grounded = (
            await client.post(
                "/ask", json={"query": "What is water composed of?"}, headers=headers
            )
        ).json()
        assert grounded["grounded"] is True
        assert grounded["user_id"] == USER
        assert any("water.md" in c["source"] for c in grounded["citations"])

        refused = (
            await client.post(
                "/ask",
                json={"query": "What is the capital of France?"},
                headers=headers,
            )
        ).json()
        assert refused["grounded"] is False
        assert refused["answer"] == REFUSAL_TEXT


async def test_api_reports_unsupported_file_as_per_file_failure():
    settings = Settings()
    pipe = build_fake_pipeline(settings)

    async with _async_client(create_app(pipeline=pipe)) as client:
        r = await client.post("/ask", json={"query": ""})  # empty query rejected by schema
        assert r.status_code == 422

        # An unsupported file no longer fails the whole batch: it is reported as
        # a per-file failure (structured code, no prose) in a 200 response.
        bad = await client.post(
            "/ingest", files={"files": ("data.csv", b"a,b,c", "text/csv")}
        )
        assert bad.status_code == 200
        body = bad.json()
        assert body["documents"] == 0
        assert body["chunks"] == 0
        assert body["files"] == []
        assert len(body["failures"]) == 1
        failure = body["failures"][0]
        assert failure["code"] == "UNSUPPORTED_TYPE"
        assert "data.csv" in failure["source"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
