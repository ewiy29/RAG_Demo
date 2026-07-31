"""Vector store interface.

Retrieval depends only on this Protocol, so the backing store (in-memory for
tests, Chroma for local use, pgvector later) can be swapped without touching
ingestion, retrieval, or generation. Scores are always cosine similarity in
[-1, 1] where higher means more similar, regardless of how the concrete store
represents distance internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class QueryHit:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


@runtime_checkable
class VectorStore(Protocol):
    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        """Insert (or upsert) chunks with their vectors and metadata."""
        ...

    def query(self, embedding: Sequence[float], k: int) -> list[QueryHit]:
        """Return up to ``k`` nearest hits, sorted by descending similarity."""
        ...

    def count(self) -> int:
        ...

    def reset(self) -> None:
        """Remove all stored vectors (used between ingest runs / in tests)."""
        ...
