"""Unit tests for retrieval: ranking order and threshold behaviour."""

from __future__ import annotations

from rag.config import Settings
from rag.ingest import ingest_paths
from rag.providers.fake_provider import FakeEmbeddingProvider
from rag.retrieve import retrieve
from rag.vectorstore.in_memory import InMemoryVectorStore


def _settings(**overrides) -> Settings:
    base = dict(provider="fake", chunk_size=1000, chunk_overlap=0)
    base.update(overrides)
    return Settings(**base)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_ranks_more_relevant_chunk_first(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "cats.md", "Cats are small domesticated feline animals.")
    _write(tmp_path, "rockets.md", "Rockets use combustion to reach orbit in space.")
    ingest_paths([tmp_path], store, embedder, _settings())

    results = retrieve(
        "Tell me about domesticated cats and felines",
        store,
        embedder,
        k=5,
        min_score=0.0,
    )
    assert results, "expected at least one hit"
    assert "cats.md" in results[0].source
    # Scores are sorted descending.
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))


def test_threshold_excludes_low_similarity_hits(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "cats.md", "Cats are small domesticated feline animals.")
    ingest_paths([tmp_path], store, embedder, _settings())

    # A totally unrelated query should not clear a modest threshold.
    off_topic = retrieve(
        "quarterly financial derivatives regulation",
        store,
        embedder,
        k=5,
        min_score=0.2,
    )
    assert off_topic == []

    # The same query with threshold 0 still returns something (ranking only).
    lenient = retrieve(
        "quarterly financial derivatives regulation",
        store,
        embedder,
        k=5,
        min_score=0.0,
    )
    assert len(lenient) >= 1


def test_k_limits_number_of_results(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    text = " ".join(f"topic{i} sentence about subject {i}." for i in range(10))
    _write(tmp_path, "many.md", text)
    ingest_paths([tmp_path], store, embedder, _settings(chunk_size=30, chunk_overlap=0))

    results = retrieve("subject", store, embedder, k=2, min_score=0.0)
    assert len(results) <= 2


def test_metadata_roundtrips_source_and_index(tmp_path):
    embedder = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    _write(tmp_path, "doc.md", "alpha beta gamma delta epsilon zeta")
    ingest_paths([tmp_path], store, embedder, _settings(chunk_size=15, chunk_overlap=0))

    results = retrieve("alpha", store, embedder, k=5, min_score=0.0)
    assert results
    top = results[0]
    assert "doc.md" in top.source
    assert top.chunk_index >= 0
    assert top.id == f"{top.source}#{top.chunk_index}"
