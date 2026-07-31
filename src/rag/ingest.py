"""Ingestion pipeline: load -> chunk -> embed -> store.

Chunk ids are deterministic (``{source}#{chunk_index}``) so re-ingesting the
same corpus upserts rather than duplicates. Embeddings are computed in a single
batched call per ingest run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_document
from .config import Settings
from .documents import load_paths
from .logging_utils import get_logger
from .providers.base import EmbeddingProvider
from .vectorstore.base import VectorStore

logger = get_logger("rag.ingest")


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int


def _chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source}#{chunk.chunk_index}"


def ingest_paths(
    paths: list[str] | list[Path],
    store: VectorStore,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> IngestResult:
    """Ingest files/directories into the vector store.

    Returns counts of documents and chunks written.
    """

    documents = load_paths(paths)

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(doc, size=settings.chunk_size, overlap=settings.chunk_overlap)
        )

    if not chunks:
        logger.info("ingest_complete", extra={"documents": len(documents), "chunks": 0})
        return IngestResult(documents=len(documents), chunks=0)

    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    ids = [_chunk_id(c) for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    logger.info(
        "ingest_complete",
        extra={"documents": len(documents), "chunks": len(chunks)},
    )
    return IngestResult(documents=len(documents), chunks=len(chunks))
