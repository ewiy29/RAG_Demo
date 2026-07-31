"""Chroma-backed vector store (default for local/production use).

Chosen because it runs locally with zero external setup and persists to disk.
We pass our own embeddings in explicitly (no Chroma embedding function), keeping
embedding generation owned by our provider layer. The collection uses cosine
space, and we convert Chroma's cosine *distance* back into a similarity score so
the rest of the app only ever sees "higher = more similar".

Backend failures are translated into the typed ``StoreError`` taxonomy so they
reach the API as a structured envelope instead of a raw chromadb exception:
construction failures -> ``UNAVAILABLE``, writes/deletes -> ``WRITE_FAILED``,
queries -> ``QUERY_FAILED``.

chromadb's client is synchronous/blocking, so every collection call is offloaded
to a threadpool with ``asyncio.to_thread`` to keep the event loop responsive.
The ``PersistentClient`` itself is constructed synchronously in ``__init__``,
which runs once at startup (outside any request), not per call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from ..errors import StoreError, StoreErrorCode
from .base import ChunkMetadata, ChunkRecord, QueryHit


def _to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate our simple equality filter into Chroma's ``where`` form.

    A single key passes through as ``{key: value}``; multiple keys are combined
    with ``$and`` since Chroma does not accept several top-level fields.
    """

    if not where:
        return None
    if len(where) == 1:
        return dict(where)
    return {"$and": [{key: value} for key, value in where.items()]}


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection: str) -> None:
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection_name = collection
            self._collection = self._client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:  # chromadb import/construction failure
            raise StoreError(
                StoreErrorCode.UNAVAILABLE,
                context={"persist_dir": persist_dir, "collection": collection},
                message=f"failed to open Chroma store: {exc}",
            ) from exc

    async def add(self, records: Sequence[ChunkRecord]) -> None:
        if not records:
            return
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[r.id for r in records],
                embeddings=[list(r.embedding) for r in records],
                documents=[r.text for r in records],
                metadatas=[r.metadata.to_dict() for r in records],
            )
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.WRITE_FAILED,
                context={"operation": "add", "count": len(records)},
                message=f"Chroma upsert failed: {exc}",
            ) from exc

    async def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        if k <= 0:
            return []
        try:
            result = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[list(embedding)],
                n_results=k,
                where=_to_chroma_where(where),
                include=["documents", "metadatas", "distances"],
            )
            ids = result.get("ids", [[]])[0]
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            hits: list[QueryHit] = []
            for id_, doc, meta, dist in zip(ids, docs, metas, dists):
                # Cosine distance in [0, 2]; similarity = 1 - distance in [-1, 1].
                hits.append(
                    QueryHit(
                        id=id_,
                        text=doc or "",
                        metadata=ChunkMetadata.from_dict(dict(meta or {})),
                        score=1.0 - float(dist),
                    )
                )
            return hits
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.QUERY_FAILED,
                context={"operation": "query", "k": k},
                message=f"Chroma query failed: {exc}",
            ) from exc

    async def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        try:
            await asyncio.to_thread(self._collection.delete, ids=list(ids))
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.WRITE_FAILED,
                context={"operation": "delete", "count": len(ids)},
                message=f"Chroma delete failed: {exc}",
            ) from exc

    async def delete_by_source(
        self, source: str, *, user_id: str | None = None
    ) -> None:
        where: dict[str, Any] = {"source": source}
        if user_id is not None:
            where["user_id"] = user_id
        try:
            await asyncio.to_thread(
                self._collection.delete, where=_to_chroma_where(where)
            )
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.WRITE_FAILED,
                context={
                    "operation": "delete_by_source",
                    "source": source,
                    "user_id": user_id,
                },
                message=f"Chroma delete_by_source failed: {exc}",
            ) from exc

    async def delete_by_user(self, user_id: str) -> None:
        try:
            await asyncio.to_thread(
                self._collection.delete, where={"user_id": user_id}
            )
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.WRITE_FAILED,
                context={"operation": "delete_by_user", "user_id": user_id},
                message=f"Chroma delete_by_user failed: {exc}",
            ) from exc

    async def delete_expired(self, cutoff_epoch: float) -> None:
        # ``$lt`` is a native Chroma range operator, passed through directly
        # rather than via the equality translator.
        try:
            await asyncio.to_thread(
                self._collection.delete,
                where={"created_at": {"$lt": cutoff_epoch}},
            )
        except Exception as exc:
            raise StoreError(
                StoreErrorCode.WRITE_FAILED,
                context={"operation": "delete_expired", "cutoff": cutoff_epoch},
                message=f"Chroma delete_expired failed: {exc}",
            ) from exc

    async def count(self) -> int:
        return await asyncio.to_thread(self._collection.count)

    async def reset(self) -> None:
        def _reset() -> None:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        await asyncio.to_thread(_reset)
