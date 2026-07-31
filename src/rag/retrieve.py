"""Retrieval: embed the query, fetch top-k, apply a minimum-score threshold.

The threshold is the guardrail's first line of defence: if no chunk is at least
``min_score`` similar to the query, retrieval returns an empty list and the
generator refuses to answer rather than grounding on irrelevant text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .providers.base import EmbeddingProvider
from .vectorstore.base import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float


def retrieve(
    query: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    k: int,
    min_score: float,
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks scoring at least ``min_score``, best first."""

    query_embedding = embedder.embed([query])[0]
    hits = store.query(query_embedding, k=k)

    results: list[RetrievedChunk] = []
    for hit in hits:
        if hit.score < min_score:
            continue
        results.append(
            RetrievedChunk(
                id=hit.id,
                text=hit.text,
                source=str(hit.metadata.get("source", "")),
                chunk_index=int(hit.metadata.get("chunk_index", 0)),
                score=hit.score,
            )
        )
    return results
