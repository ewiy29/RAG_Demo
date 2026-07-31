"""Chroma-backed vector store (default for local/production use).

Chosen because it runs locally with zero external setup and persists to disk.
We pass our own embeddings in explicitly (no Chroma embedding function), keeping
embedding generation owned by our provider layer. The collection uses cosine
space, and we convert Chroma's cosine *distance* back into a similarity score so
the rest of the app only ever sees "higher = more similar".
"""

from __future__ import annotations

from typing import Sequence

from .base import QueryHit


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[dict(m) for m in metadatas],
        )

    def query(self, embedding: Sequence[float], k: int) -> list[QueryHit]:
        if k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
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
                    metadata=dict(meta or {}),
                    score=1.0 - float(dist),
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
