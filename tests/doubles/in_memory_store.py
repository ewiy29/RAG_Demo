"""A dependency-free, in-memory vector store -- a TEST DOUBLE, not shipped.

Relocated under ``tests/`` (WS10): the production factory
(``rag.vectorstore.build_store``) only builds Chroma; tests inject this store
directly. It remains the reference implementation of the store *contract*:
upsert-by-id, consistent embedding dimensionality, delete/delete-by-source, and
metadata filtering on query. Similarity is exact cosine similarity, so it is a
faithful reference for the threshold behaviour that the Chroma store
approximates with ANN search.

Concurrency: the methods are ``async`` to satisfy the interface, but their
bodies do only synchronous, non-``await``ing work, so each call runs to
completion atomically on the single-threaded event loop -- concurrent requests
cannot interleave a partial dict mutation. That atomicity (rather than a lock)
is what makes the shared store safe under the async server. It is still not a
server-grade store (unshared, in-process, lost on restart).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from rag.vectorstore.base import ChunkMetadata, ChunkRecord, QueryHit


@dataclass
class _Record:
    id: str
    embedding: list[float]
    text: str
    metadata: ChunkMetadata


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _matches(metadata: ChunkMetadata, where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    stored = metadata.to_dict()
    return all(stored.get(key) == value for key, value in where.items())


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}
        # The one embedding dimension every stored/query vector must share.
        # Set on the first insert; enforced thereafter to avoid a silent
        # cosine over a truncated vector prefix.
        self._dim: int | None = None

    def _check_dim(self, embedding: Sequence[float]) -> None:
        if self._dim is None:
            self._dim = len(embedding)
        elif len(embedding) != self._dim:
            raise ValueError(
                f"embedding dimension {len(embedding)} does not match store "
                f"dimension {self._dim}"
            )

    async def add(self, records: Sequence[ChunkRecord]) -> None:
        for record in records:
            self._check_dim(record.embedding)
        # Upsert by id: a repeated id replaces the prior record in place.
        for record in records:
            self._records[record.id] = _Record(
                id=record.id,
                embedding=list(record.embedding),
                text=record.text,
                metadata=record.metadata,
            )

    async def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        if k <= 0:
            return []
        if self._dim is not None and len(embedding) != self._dim:
            raise ValueError(
                f"query embedding dimension {len(embedding)} does not match "
                f"store dimension {self._dim}"
            )
        scored = [
            QueryHit(
                id=rec.id,
                text=rec.text,
                metadata=rec.metadata,
                score=_cosine(embedding, rec.embedding),
            )
            for rec in self._records.values()
            if _matches(rec.metadata, where)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    async def delete(self, ids: Sequence[str]) -> None:
        for id_ in ids:
            self._records.pop(id_, None)

    async def delete_by_source(
        self, source: str, *, user_id: str | None = None
    ) -> None:
        doomed = [
            id_
            for id_, rec in self._records.items()
            if rec.metadata.source == source
            and (user_id is None or rec.metadata.user_id == user_id)
        ]
        for id_ in doomed:
            del self._records[id_]

    async def delete_by_user(self, user_id: str) -> None:
        doomed = [
            id_
            for id_, rec in self._records.items()
            if rec.metadata.user_id == user_id
        ]
        for id_ in doomed:
            del self._records[id_]

    async def delete_expired(self, cutoff_epoch: float) -> None:
        doomed = [
            id_
            for id_, rec in self._records.items()
            if rec.metadata.created_at < cutoff_epoch
        ]
        for id_ in doomed:
            del self._records[id_]

    async def count(self) -> int:
        return len(self._records)

    async def reset(self) -> None:
        self._records.clear()
        self._dim = None
