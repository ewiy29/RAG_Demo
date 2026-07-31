"""WS6 tenancy tests: per-user isolation, minted GUID echo, purge, TTL cleanup.

Runs fully offline against the fake provider + in-memory store. Exercises the
tenant boundary end-to-end through the HTTP API (one user cannot retrieve
another's uploads) plus the pipeline-level purge and on-demand TTL cleanup
helpers.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
from httpx import ASGITransport

from doubles import FakeEmbeddingProvider, InMemoryVectorStore, build_fake_pipeline
from rag.api import create_app
from rag.config import Settings
from rag.ingest import ingest_paths
from rag.retrieve import retrieve

FIXTURES = Path(__file__).parent / "fixtures"


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _settings(**overrides) -> Settings:
    base = dict(top_k=4, min_score=0.2)
    base.update(overrides)
    return Settings(**base)


async def test_api_ingest_mints_and_echoes_user_id():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        # No X-User-Id header -> the server mints one and echoes it back.
        with open(FIXTURES / "water.md", "rb") as fh:
            r = await client.post(
                "/ingest", files={"files": ("water.md", fh.read(), "text/markdown")}
            )
        assert r.status_code == 200
        minted = r.json()["user_id"]
        assert minted
        assert r.headers["X-User-Id"] == minted


async def test_api_isolation_between_users():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        # User A uploads the water fixture.
        with open(FIXTURES / "water.md", "rb") as fh:
            await client.post(
                "/ingest",
                files={"files": ("water.md", fh.read(), "text/markdown")},
                headers={"X-User-Id": "user-a"},
            )

        # User A can retrieve/ground on it.
        a = (
            await client.post(
                "/ask",
                json={"query": "What is water composed of?"},
                headers={"X-User-Id": "user-a"},
            )
        ).json()
        assert a["grounded"] is True

        # User B, who uploaded nothing, gets a refusal (isolation).
        b = (
            await client.post(
                "/ask",
                json={"query": "What is water composed of?"},
                headers={"X-User-Id": "user-b"},
            )
        ).json()
        assert b["grounded"] is False
        assert b["retrieved"] == []


async def test_api_purge_deletes_only_requesting_user():
    pipe = build_fake_pipeline(_settings())

    async with _async_client(create_app(pipeline=pipe)) as client:
        for user in ("user-a", "user-b"):
            with open(FIXTURES / "water.md", "rb") as fh:
                await client.post(
                    "/ingest",
                    files={"files": ("water.md", fh.read(), "text/markdown")},
                    headers={"X-User-Id": user},
                )

        # Purge user A's data.
        purged = await client.request(
            "DELETE", "/documents", headers={"X-User-Id": "user-a"}
        )
        assert purged.status_code == 200
        assert purged.json() == {"user_id": "user-a", "status": "purged"}

        # User A now refuses (data gone)...
        a = (
            await client.post(
                "/ask",
                json={"query": "What is water composed of?"},
                headers={"X-User-Id": "user-a"},
            )
        ).json()
        assert a["grounded"] is False

        # ...but user B is untouched.
        b = (
            await client.post(
                "/ask",
                json={"query": "What is water composed of?"},
                headers={"X-User-Id": "user-b"},
            )
        ).json()
        assert b["grounded"] is True


async def test_pipeline_cleanup_expired_removes_stale_user_data(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    (tmp_path / "notes.md").write_text("water is made of hydrogen and oxygen", encoding="utf-8")

    settings = _settings(chunk_size=1000, chunk_overlap=0, session_ttl_seconds=1)
    await ingest_paths([tmp_path / "notes.md"], store, embedder, settings, user_id="ua")
    assert await store.count() >= 1

    pipe = build_fake_pipeline(settings)
    pipe.store = store  # drive cleanup against the store we just populated

    # A negative TTL makes every existing record already expired.
    await pipe.cleanup_expired(ttl_seconds=-1)
    assert await store.count() == 0

    # And the user can no longer retrieve anything.
    hits = await retrieve("water", store, embedder, k=5, min_score=0.0, user_id="ua")
    assert hits == []


async def test_pipeline_cleanup_expired_keeps_fresh_data(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    (tmp_path / "notes.md").write_text("water is made of hydrogen and oxygen", encoding="utf-8")

    settings = _settings(chunk_size=1000, chunk_overlap=0)
    await ingest_paths([tmp_path / "notes.md"], store, embedder, settings, user_id="ua")
    fresh_count = await store.count()
    assert fresh_count >= 1

    pipe = build_fake_pipeline(settings)
    pipe.store = store

    # A large TTL means nothing just-ingested is expired yet.
    assert time.time() > 0  # sanity: real clock drives created_at
    await pipe.cleanup_expired(ttl_seconds=10_000)
    assert await store.count() == fresh_count
