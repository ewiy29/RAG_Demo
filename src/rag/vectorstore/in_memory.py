"""A dependency-free, in-memory vector store.

Used as the default in tests (fast, deterministic, no external process) and
handy for quick local experiments. Similarity is exact cosine similarity, so
it is a faithful reference implementation for the threshold behaviour that the
Chroma store approximates with ANN search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .base import QueryHit


@dataclass
class _Record:
    id: str
    embedding: list[float]
    text: str
    metadata: dict


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("add() requires ids/embeddings/documents/metadatas of equal length")
        for id_, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._records[id_] = _Record(id_, list(emb), doc, dict(meta))

    def query(self, embedding: Sequence[float], k: int) -> list[QueryHit]:
        scored = [
            QueryHit(
                id=rec.id,
                text=rec.text,
                metadata=rec.metadata,
                score=_cosine(embedding, rec.embedding),
            )
            for rec in self._records.values()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: max(k, 0)]

    def count(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        self._records.clear()
