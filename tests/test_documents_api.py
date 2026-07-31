"""Document-management API tests: list sources + per-file delete.

Covers the endpoints the manage-files UI drives (``GET /documents`` and
``DELETE /documents/{source}``): distinct sources with chunk counts, per-user
isolation, and that a per-file delete removes only the named source (leaving
the user's other files and other users untouched). Runs fully offline against
the fake provider + in-memory store via ``build_fake_pipeline``.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from httpx import ASGITransport

from doubles import build_fake_pipeline
from rag.api import create_app
from rag.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _settings(**overrides) -> Settings:
    base = dict(top_k=4, min_score=0.2)
    base.update(overrides)
    return Settings(**base)


async def _ingest(client, filename: str, user: str) -> None:
    with open(FIXTURES / filename, "rb") as fh:
        r = await client.post(
            "/ingest",
            files={"files": (filename, fh.read(), "text/markdown")},
            headers={"X-User-Id": user},
        )
    assert r.status_code == 200


async def test_list_documents_returns_sources_with_chunk_counts():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        await _ingest(client, "water.md", "user-a")
        await _ingest(client, "photosynthesis.md", "user-a")

        r = await client.get("/documents", headers={"X-User-Id": "user-a"})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == "user-a"

        docs = {d["source"]: d["chunks"] for d in body["documents"]}
        assert set(docs) == {"water.md", "photosynthesis.md"}
        # Every listed source must have at least one stored chunk.
        assert all(count >= 1 for count in docs.values())


async def test_list_documents_is_per_user_and_empty_by_default():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        await _ingest(client, "water.md", "user-a")

        # Another user who uploaded nothing sees an empty list (isolation).
        r = await client.get("/documents", headers={"X-User-Id": "user-b"})
        assert r.status_code == 200
        assert r.json()["documents"] == []


async def test_delete_single_document_removes_only_that_source():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        await _ingest(client, "water.md", "user-a")
        await _ingest(client, "photosynthesis.md", "user-a")

        deleted = await client.request(
            "DELETE", "/documents/water.md", headers={"X-User-Id": "user-a"}
        )
        assert deleted.status_code == 200
        assert deleted.json() == {
            "user_id": "user-a",
            "source": "water.md",
            "status": "deleted",
        }

        # Only water.md is gone; photosynthesis.md remains.
        remaining = await client.get("/documents", headers={"X-User-Id": "user-a"})
        sources = {d["source"] for d in remaining.json()["documents"]}
        assert sources == {"photosynthesis.md"}


async def test_delete_single_document_only_affects_requesting_user():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        await _ingest(client, "water.md", "user-a")
        await _ingest(client, "water.md", "user-b")

        await client.request(
            "DELETE", "/documents/water.md", headers={"X-User-Id": "user-a"}
        )

        # User A's copy is gone...
        a = await client.get("/documents", headers={"X-User-Id": "user-a"})
        assert a.json()["documents"] == []

        # ...but user B's same-named file is untouched.
        b = await client.get("/documents", headers={"X-User-Id": "user-b"})
        assert {d["source"] for d in b.json()["documents"]} == {"water.md"}


async def test_delete_missing_document_is_idempotent():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        r = await client.request(
            "DELETE", "/documents/nope.md", headers={"X-User-Id": "user-a"}
        )
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
