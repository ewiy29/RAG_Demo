"""WS10 API hardening: health disclosure, upload size limit, CORS.

Runs fully offline against the fake provider + in-memory stores injected via
``doubles.build_fake_pipeline``.
"""

from __future__ import annotations

import pathlib

import httpx
from httpx import ASGITransport

from doubles import build_fake_pipeline
from rag.api import create_app
from rag.config import Settings

FIXTURES_USER = "hardening-user"


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_does_not_disclose_provider():
    pipe = build_fake_pipeline(Settings())
    async with _async_client(create_app(pipeline=pipe)) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok"}
    assert "provider" not in body


async def test_oversized_upload_is_rejected_per_file():
    # A tiny cap so a small payload trips the limit.
    settings = Settings(max_upload_bytes=16)
    pipe = build_fake_pipeline(settings)

    async with _async_client(create_app(pipeline=pipe)) as client:
        big = b"x" * 1024  # well over the 16-byte cap
        resp = await client.post(
            "/ingest",
            files={"files": ("big.md", big, "text/markdown")},
            headers={"X-User-Id": FIXTURES_USER},
        )

    assert resp.status_code == 200  # partial-success contract, not a hard 4xx
    body = resp.json()
    assert body["chunks"] == 0
    assert body["files"] == []
    assert len(body["failures"]) == 1
    failure = body["failures"][0]
    assert failure["code"] == "TOO_LARGE"
    assert failure["context"]["max_bytes"] == 16


async def test_within_limit_upload_still_ingests():
    settings = Settings(max_upload_bytes=10_000)
    pipe = build_fake_pipeline(settings)

    async with _async_client(create_app(pipeline=pipe)) as client:
        resp = await client.post(
            "/ingest",
            files={"files": ("notes.md", b"water is hydrogen and oxygen", "text/markdown")},
            headers={"X-User-Id": FIXTURES_USER},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["failures"] == []
    assert body["chunks"] >= 1


async def test_ingest_writes_nothing_to_disk(monkeypatch, tmp_path):
    # No-persistence contract: /ingest must process uploads entirely in memory.
    # Fail loudly if anything tries to stage a raw upload on disk.
    def _no_writes(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError(f"unexpected disk write to {self}")

    monkeypatch.setattr(pathlib.Path, "write_bytes", _no_writes)
    monkeypatch.chdir(tmp_path)  # so any stray relative write lands here

    pipe = build_fake_pipeline(Settings())
    async with _async_client(create_app(pipeline=pipe)) as client:
        resp = await client.post(
            "/ingest",
            files={"files": ("notes.md", b"water is hydrogen and oxygen", "text/markdown")},
            headers={"X-User-Id": FIXTURES_USER},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["failures"] == []
    assert body["chunks"] >= 1
    assert body["files"] == ["notes.md"]
    # Nothing was staged: no uploads/ directory was created anywhere.
    assert not (tmp_path / "uploads").exists()


async def test_ingest_still_reports_unsupported_and_oversized_failures():
    # A cap that admits the short valid file but rejects the padded one.
    settings = Settings(max_upload_bytes=64)
    pipe = build_fake_pipeline(settings)

    async with _async_client(create_app(pipeline=pipe)) as client:
        resp = await client.post(
            "/ingest",
            files=[
                ("files", ("data.csv", b"a,b,c", "text/csv")),
                ("files", ("big.md", b"x" * 1024, "text/markdown")),
                ("files", ("ok.md", b"water is hydrogen", "text/markdown")),
            ],
            headers={"X-User-Id": FIXTURES_USER},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["files"] == ["ok.md"]
    codes = {f["source"]: f["code"] for f in body["failures"]}
    assert codes["data.csv"] == "UNSUPPORTED_TYPE"
    assert codes["big.md"] == "TOO_LARGE"
    assert body["chunks"] >= 1


async def test_cors_headers_present_for_browser_origin():
    pipe = build_fake_pipeline(Settings())
    async with _async_client(create_app(pipeline=pipe)) as client:
        # A CORS preflight from a browser origin should be answered with the
        # allow-origin header so the React UI can call the API.
        resp = await client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert resp.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")
