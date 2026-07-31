"""Ingestion pipeline: load -> chunk -> embed -> store.

Chunk ids are deterministic (``{source}#{chunk_index}``) so re-ingesting the
same corpus upserts rather than duplicates. Embeddings are computed in a single
batched call per ingest run.

Ingestion is **per-user**: a ``user_id`` (a minted tenant GUID, not auth) is
stamped on every chunk's metadata, folded into the chunk id
(``{user_id}/{source}#{chunk_index}``) so two users' same-named files never
collide, and used to scope the delete-before-add so re-ingesting one user's
file never touches another user's copy. Each chunk is also stamped with a
``created_at`` timestamp for TTL cleanup.

Each successfully loaded source is **deleted from the store before its new
chunks are added**, so editing one file and re-ingesting it cannot leave stale
higher-index chunks (``source#4``, ``source#5``, ...) behind when the new
version produces fewer chunks. Upsert alone could not remove those orphans.

Loading is partial-success: files that fail to load (missing, unsupported,
undecodable, empty) are reported as ``failures`` on the result rather than
aborting the whole run, so a multi-file upload can ingest what it can and tell
the caller which files failed and why. A source that fails to load keeps its
previously-ingested chunks (we do not delete what we could not replace).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_document
from .config import Settings
from .documents import (
    Document,
    FileFailure,
    load_paths_with_results,
    load_uploads_with_results,
)
from .logging_utils import get_logger
from .providers.base import EmbeddingProvider
from .vectorstore.base import ChunkMetadata, ChunkRecord, VectorStore

logger = get_logger("rag.ingest")


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    failures: list[FileFailure] = field(default_factory=list)


def _chunk_id(chunk: Chunk, user_id: str) -> str:
    return f"{user_id}/{chunk.source}#{chunk.chunk_index}"


def _chunk_all(documents: list[Document], settings: Settings) -> list[Chunk]:
    """Split every loaded document into chunks (CPU-bound; run off the loop)."""

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(doc, size=settings.chunk_size, overlap=settings.chunk_overlap)
        )
    return chunks


async def ingest_documents(
    documents: list[Document],
    store: VectorStore,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    user_id: str,
    failures: list[FileFailure] | None = None,
    start: float | None = None,
) -> IngestResult:
    """Chunk -> embed -> store a batch of already-loaded documents for one user.

    The shared post-load stage both the path- and upload-based entry points
    funnel through, so tenancy (``user_id``/``created_at`` stamping, chunk-id
    scheme, delete-before-readd scoping) and the ``ingest_complete`` log stay
    identical regardless of where the documents came from. ``failures`` carries
    any per-file load failures through onto the result; ``start`` lets the
    caller include load time in the reported ``duration_ms``.
    """

    if start is None:
        start = time.perf_counter()
    failures = failures if failures is not None else []

    # Clear each loaded source's prior chunks (scoped to this user) before
    # re-adding so a shrunk re-ingest cannot orphan stale higher-index chunks.
    # Sources that failed to load are absent here, so their existing chunks are
    # preserved; another user's same-named file is untouched.
    for source in dict.fromkeys(doc.source for doc in documents):
        await store.delete_by_source(source, user_id=user_id)

    # Chunking is CPU-bound; offload it so the event loop stays responsive.
    chunks = await asyncio.to_thread(_chunk_all, documents, settings)

    if not chunks:
        logger.info(
            "ingest_complete",
            extra={
                "user_id": user_id,
                "documents": len(documents),
                "chunks": 0,
                "failures": len(failures),
                "duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
            },
        )
        return IngestResult(documents=len(documents), chunks=0, failures=failures)

    created_at = time.time()
    texts = [c.text for c in chunks]
    embeddings = await embedder.embed(texts)
    records = [
        ChunkRecord(
            id=_chunk_id(chunk, user_id),
            embedding=embedding,
            text=chunk.text,
            metadata=ChunkMetadata(
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                user_id=user_id,
                created_at=created_at,
            ),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await store.add(records)

    logger.info(
        "ingest_complete",
        extra={
            "user_id": user_id,
            "documents": len(documents),
            "chunks": len(chunks),
            "failures": len(failures),
            "duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
        },
    )
    return IngestResult(
        documents=len(documents), chunks=len(chunks), failures=failures
    )


async def ingest_paths(
    paths: list[str] | list[Path],
    store: VectorStore,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    user_id: str,
) -> IngestResult:
    """Ingest files/directories into the vector store for one user.

    Loads from disk (corpus folders + tests still use paths), then hands the
    loaded documents to :func:`ingest_documents`. Every chunk is stamped with
    ``user_id`` (tenant isolation) and a ``created_at`` timestamp (TTL
    cleanup). Returns counts of documents and chunks written, plus any per-file
    load failures. Files that fail to load do not stop the run.
    """

    start = time.perf_counter()

    # Document loading is blocking file I/O; offload it so the event loop stays
    # responsive during a large ingest.
    load_result = await asyncio.to_thread(load_paths_with_results, paths)
    return await ingest_documents(
        load_result.documents,
        store,
        embedder,
        settings,
        user_id=user_id,
        failures=load_result.failures,
        start=start,
    )


async def ingest_uploads(
    uploads: list[tuple[str, bytes]],
    store: VectorStore,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    user_id: str,
) -> IngestResult:
    """Ingest in-memory uploads (``(filename, data)``) for one user.

    The no-persistence path: uploaded bytes are normalised, chunked, embedded,
    and stored straight from memory -- the raw upload is never written to disk.
    Loading is partial-success (a bad file lands in ``failures`` while the rest
    ingest), then the loaded documents go through the same
    :func:`ingest_documents` stage as the path loader.
    """

    start = time.perf_counter()

    # Decoding/PDF extraction is CPU-bound; offload it to keep the loop free.
    load_result = await asyncio.to_thread(load_uploads_with_results, uploads)
    return await ingest_documents(
        load_result.documents,
        store,
        embedder,
        settings,
        user_id=user_id,
        failures=load_result.failures,
        start=start,
    )
