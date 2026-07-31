"""Vector store selection.

``build_store`` returns a concrete store based on settings. Chroma is the only
shipped backend; it is imported lazily so the package imports cleanly even if
chromadb isn't installed. The in-memory store is a test double (a reference
implementation of the store contract) and lives under ``tests/doubles`` (WS10);
tests inject it directly rather than selecting it via a ``kind`` string.
"""

from __future__ import annotations

from ..config import Settings
from .base import ChunkMetadata, ChunkRecord, QueryHit, VectorStore

__all__ = [
    "ChunkMetadata",
    "ChunkRecord",
    "QueryHit",
    "VectorStore",
    "build_store",
]


def build_store(settings: Settings, kind: str = "chroma") -> VectorStore:
    kind = kind.lower()
    if kind == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=settings.persist_dir, collection=settings.collection
        )
    raise ValueError(f"Unknown vector store kind {kind!r}. Use 'chroma'.")
