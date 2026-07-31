"""Unit tests for retrieval: ranking order and threshold behaviour."""

from __future__ import annotations

import itertools

from doubles import FakeEmbeddingProvider, InMemoryVectorStore
from rag.config import Settings
from rag.ingest import ingest_paths
from rag.retrieve import retrieve

USER = "user-under-test"


def _settings(**overrides) -> Settings:
    base = {"chunk_size": 1000, "chunk_overlap": 0}
    base.update(overrides)
    return Settings(**base)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


async def test_ranks_more_relevant_chunk_first(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "cats.md", "Cats are small domesticated feline animals.")
    _write(tmp_path, "rockets.md", "Rockets use combustion to reach orbit in space.")
    await ingest_paths([tmp_path], store, embedder, _settings(), user_id=USER)

    results = await retrieve(
        "Tell me about domesticated cats and felines",
        store,
        embedder,
        k=5,
        min_score=0.0,
        user_id=USER,
    )
    assert results, "expected at least one hit"
    assert "cats.md" in results[0].source
    # Scores are sorted descending.
    assert all(a.score >= b.score for a, b in itertools.pairwise(results))


async def test_threshold_excludes_low_similarity_hits(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "cats.md", "Cats are small domesticated feline animals.")
    await ingest_paths([tmp_path], store, embedder, _settings(), user_id=USER)

    # A totally unrelated query should not clear a modest threshold.
    off_topic = await retrieve(
        "quarterly financial derivatives regulation",
        store,
        embedder,
        k=5,
        min_score=0.2,
        user_id=USER,
    )
    assert off_topic == []

    # The same query with threshold 0 still returns something (ranking only).
    lenient = await retrieve(
        "quarterly financial derivatives regulation",
        store,
        embedder,
        k=5,
        min_score=0.0,
        user_id=USER,
    )
    assert len(lenient) >= 1


async def test_k_limits_number_of_results(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    text = " ".join(f"topic{i} sentence about subject {i}." for i in range(10))
    _write(tmp_path, "many.md", text)
    await ingest_paths(
        [tmp_path], store, embedder, _settings(chunk_size=30, chunk_overlap=0), user_id=USER
    )

    results = await retrieve("subject", store, embedder, k=2, min_score=0.0, user_id=USER)
    assert len(results) <= 2


async def test_reingest_of_shrunk_file_leaves_no_stale_chunks(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    path = _write(
        tmp_path,
        "doc.md",
        " ".join(f"word{i} filler text here" for i in range(10)),
    )
    await ingest_paths(
        [path], store, embedder, _settings(chunk_size=30, chunk_overlap=0), user_id=USER
    )
    first_count = await store.count()
    assert first_count > 1, "expected the long version to produce several chunks"

    # Edit the file down to a single short chunk and re-ingest just that file.
    path.write_text("only one short line now", encoding="utf-8")
    await ingest_paths(
        [path], store, embedder, _settings(chunk_size=30, chunk_overlap=0), user_id=USER
    )

    # delete-by-source-before-add must have removed the old higher-index chunks.
    assert await store.count() == 1
    hits = await store.query((await embedder.embed(["x"]))[0], k=50)
    assert len(hits) == 1
    assert hits[0].metadata.chunk_index == 0
    # No stale ``source#1``, ``source#2``, ... survive from the longer version.
    assert not any(hit.id.endswith("#1") for hit in hits)


async def test_metadata_roundtrips_source_and_index(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "doc.md", "alpha beta gamma delta epsilon zeta")
    await ingest_paths(
        [tmp_path], store, embedder, _settings(chunk_size=15, chunk_overlap=0), user_id=USER
    )

    results = await retrieve("alpha", store, embedder, k=5, min_score=0.0, user_id=USER)
    assert results
    top = results[0]
    assert "doc.md" in top.source
    assert top.chunk_index >= 0
    # The chunk id is now scoped by user: ``{user_id}/{source}#{chunk_index}``.
    assert top.id == f"{USER}/{top.source}#{top.chunk_index}"


async def test_retrieval_is_isolated_between_users(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "cats.md", "Cats are small domesticated feline animals.")
    # Only user A ingests the corpus.
    await ingest_paths([tmp_path], store, embedder, _settings(), user_id="user-a")

    # User A sees the chunk...
    a_hits = await retrieve(
        "domesticated cats", store, embedder, k=5, min_score=0.0, user_id="user-a"
    )
    assert a_hits, "user A should retrieve their own ingested chunk"

    # ...but user B, who ingested nothing, sees nothing (isolation).
    b_hits = await retrieve(
        "domesticated cats", store, embedder, k=5, min_score=0.0, user_id="user-b"
    )
    assert b_hits == []


async def test_reingest_scoped_to_user_does_not_clobber_other_user(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    # Both users upload a same-named file with different content.
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "notes.md").write_text("alpha content for user a", encoding="utf-8")
    (b_dir / "notes.md").write_text("beta content for user b", encoding="utf-8")
    await ingest_paths([a_dir / "notes.md"], store, embedder, _settings(), user_id="user-a")
    await ingest_paths([b_dir / "notes.md"], store, embedder, _settings(), user_id="user-b")

    # User A re-ingests their notes.md (delete-before-add scoped to user A).
    await ingest_paths([a_dir / "notes.md"], store, embedder, _settings(), user_id="user-a")

    # User B's same-named file must survive untouched.
    b_hits = await retrieve(
        "beta content", store, embedder, k=5, min_score=0.0, user_id="user-b"
    )
    assert b_hits, "user B's notes.md must not be clobbered by user A's re-ingest"
    assert "beta content" in b_hits[0].text
