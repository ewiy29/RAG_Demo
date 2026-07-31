"""Vector store selection.

``build_store`` returns a concrete store based on settings. The in-memory store
is always available (and used by the test suite); Chroma is imported lazily so
the package imports cleanly even if chromadb isn't installed.
"""

from __future__ import annotations

from ..config import Settings
from .base import QueryHit, VectorStore
from .in_memory import InMemoryVectorStore

__all__ = [
    "QueryHit",
    "VectorStore",
    "InMemoryVectorStore",
    "build_store",
]


def build_store(settings: Settings, kind: str = "chroma") -> VectorStore:
    kind = kind.lower()
    if kind == "memory":
        return InMemoryVectorStore()
    if kind == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            persist_dir=settings.persist_dir, collection=settings.collection
        )
    raise ValueError(f"Unknown vector store kind {kind!r}. Use 'chroma' or 'memory'.")
