"""End-to-end integration test for the full ingest -> ask pipeline.

Runs entirely offline against the fake provider and an in-memory store, over a
tiny fixture corpus. Asserts both the grounded-with-citation path and the
out-of-scope refusal path, at the pipeline level and through the HTTP API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import Settings
from rag.generate import REFUSAL_TEXT
from rag.pipeline import build_pipeline

FIXTURES = Path(__file__).parent / "fixtures"


def _pipeline():
    settings = Settings(provider="fake", top_k=4, min_score=0.2, chunk_size=400, chunk_overlap=50)
    pipe = build_pipeline(settings, store_kind="memory")
    result = pipe.ingest([str(FIXTURES)])
    assert result.documents >= 2
    assert result.chunks >= 2
    return pipe


def test_in_scope_question_is_grounded_and_cites_source():
    pipe = _pipeline()
    resp = pipe.ask("What is water composed of?")

    assert resp.grounded is True
    assert resp.answer != REFUSAL_TEXT
    assert resp.citations, "a grounded answer must carry at least one citation"
    # The citation must point back to the water fixture.
    assert any("water.md" in c.source for c in resp.citations)
    # The retrieved context that won should be the water document.
    assert "water.md" in resp.retrieved[0].source


def test_out_of_scope_question_triggers_refusal():
    pipe = _pipeline()
    resp = pipe.ask("Who won the 1998 football world cup final?")

    assert resp.grounded is False
    assert resp.answer == REFUSAL_TEXT
    assert resp.citations == []
    assert resp.retrieved == []  # nothing cleared the threshold


def test_api_ingest_then_ask_grounded_and_refusal(tmp_path):
    settings = Settings(provider="fake", top_k=4, min_score=0.2, upload_dir=str(tmp_path / "uploads"))
    pipe = build_pipeline(settings, store_kind="memory")
    client = TestClient(create_app(pipeline=pipe))

    # Ingest the water fixture via multipart upload.
    with open(FIXTURES / "water.md", "rb") as fh:
        r = client.post("/ingest", files={"files": ("water.md", fh.read(), "text/markdown")})
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    grounded = client.post("/ask", json={"query": "What is water composed of?"}).json()
    assert grounded["grounded"] is True
    assert any("water.md" in c["source"] for c in grounded["citations"])

    refused = client.post("/ask", json={"query": "What is the capital of France?"}).json()
    assert refused["grounded"] is False
    assert refused["answer"] == REFUSAL_TEXT


def test_api_rejects_unsupported_file_type(tmp_path):
    settings = Settings(provider="fake", upload_dir=str(tmp_path / "uploads"))
    pipe = build_pipeline(settings, store_kind="memory")
    client = TestClient(create_app(pipeline=pipe))

    r = client.post("/ask", json={"query": ""})  # empty query rejected by schema
    assert r.status_code == 422

    bad = client.post("/ingest", files={"files": ("data.csv", b"a,b,c", "text/csv")})
    assert bad.status_code == 400


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
