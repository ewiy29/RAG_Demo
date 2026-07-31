"""Vector store interface + data model.

Retrieval and ingestion depend only on this Protocol, so the backing store
(in-memory for tests, Chroma for local use, pgvector/Cosmos later) can be
swapped without touching the rest of the app. Scores are always cosine
similarity in [-1, 1] where higher means more similar, regardless of how the
concrete store represents distance internally.

Data model
----------
Chunks cross the boundary as a single ``ChunkRecord`` bundle (id + embedding +
text + typed ``ChunkMetadata``) rather than four parallel arrays, so a store
can never silently mis-pair a vector with the wrong text/metadata. ``QueryHit``
carries the same typed ``ChunkMetadata`` back on the read side, so callers read
``hit.metadata.source`` instead of guessing untyped dict keys.

Contract every backend must honour
----------------------------------
* **Async.** Every method is a coroutine so a store backed by network I/O (a
  remote vector service) never blocks the event loop. An in-process backend
  keeps its method body synchronous (no internal ``await``), which makes each
  operation run atomically on the single-threaded event loop -- that atomicity,
  not a lock, is what makes the shared store concurrency-safe under the async
  server. A backend that wraps a blocking library offloads it to a threadpool
  (``asyncio.to_thread``).
* **Upsert by id.** ``add`` replaces any existing record with the same id in
  place (re-ingesting a changed file updates rather than duplicates).
* **Consistent dimensionality.** All stored embeddings, and every query
  embedding, share one fixed dimension; a mismatch is a caller error
  (``ValueError``), never a silently truncated cosine.
* **Deletion.** ``delete``/``delete_by_source`` remove records so a single
  source can be re-ingested or cleaned up without wiping the whole store.
* **Metadata filtering.** ``query`` accepts an optional ``where`` equality
  filter (e.g. ``{"source": ...}``) so retrieval can be scoped.
* **Tenancy.** Every chunk carries a ``user_id`` (a minted GUID, not auth) and
  a ``created_at`` epoch timestamp. Isolation is enforced by the caller passing
  ``where={"user_id": ...}`` on ``query``; ``delete_by_user`` powers "delete all
  my data" and ``delete_expired`` powers on-demand TTL cleanup of stale
  per-user data. ``delete_by_source`` can be scoped to one user so two users
  who upload a same-named file do not clobber each other on re-ingest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChunkMetadata:
    """Typed metadata stored alongside every chunk.

    The schema is explicit (rather than a loose ``dict``) because citations map
    back to a chunk by ``source`` + ``chunk_index``; formalizing it keeps that
    mapping safe. ``to_dict``/``from_dict`` bridge to backends (like Chroma)
    that persist metadata as plain dicts.

    ``user_id`` (a minted tenant GUID, not auth) and ``created_at`` (epoch
    seconds) carry the tenancy/lifecycle fields. They default so pre-tenancy
    records and terse test constructions stay valid; ingestion always stamps a
    real ``user_id`` and timestamp.
    """

    source: str
    chunk_index: int
    user_id: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "chunk_index": self.chunk_index,
            "user_id": self.user_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkMetadata:
        """Build from a stored dict, raising on a missing/invalid schema.

        A malformed metadata dict is a corruption signal, not something to
        paper over with silent defaults; the store layer turns the resulting
        error into a typed ``StoreError`` for the caller. The tenancy fields
        (``user_id``/``created_at``) are read leniently so a record written
        before tenancy still loads.
        """

        return cls(
            source=str(data["source"]),
            chunk_index=int(data["chunk_index"]),
            user_id=str(data.get("user_id", "")),
            created_at=float(data.get("created_at", 0.0)),
        )


@dataclass(frozen=True)
class ChunkRecord:
    """One chunk to store: its stable id, embedding, text, and typed metadata."""

    id: str
    embedding: Sequence[float]
    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class QueryHit:
    id: str
    text: str
    metadata: ChunkMetadata
    score: float = 0.0


@dataclass(frozen=True)
class SourceInfo:
    """One ingested source belonging to a user, plus how many chunks it holds.

    Powers the "manage my documents" UI: a user needs to see which files are
    currently referenced (and delete/re-upload them) without exposing the
    per-chunk ids. ``chunks`` is the number of stored chunks derived from the
    source, a rough proxy for its size/coverage.
    """

    source: str
    chunks: int


@runtime_checkable
class VectorStore(Protocol):
    async def add(self, records: Sequence[ChunkRecord]) -> None:
        """Upsert chunks by id (replace on duplicate id, never duplicate)."""
        ...

    async def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        where: dict[str, Any] | None = None,
    ) -> list[QueryHit]:
        """Return up to ``k`` nearest hits, sorted by descending similarity.

        ``where`` is an optional equality filter over metadata (e.g.
        ``{"source": "notes.md"}``) applied before ranking.
        """
        ...

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove records by id (no-op for ids that are not present)."""
        ...

    async def delete_by_source(
        self, source: str, *, user_id: str | None = None
    ) -> None:
        """Remove every record whose metadata ``source`` matches.

        When ``user_id`` is given, only that user's chunks for the source are
        removed, so re-ingesting one user's file never touches another user's
        same-named file.
        """
        ...

    async def delete_by_user(self, user_id: str) -> None:
        """Remove every record belonging to ``user_id`` ("delete all my data")."""
        ...

    async def list_sources(self, user_id: str) -> list[SourceInfo]:
        """Return the distinct sources ``user_id`` has ingested, with chunk counts.

        Powers the document-management UI (list what a user has uploaded so it
        can be deleted or re-uploaded). Only the caller's own sources are ever
        returned; the result is empty for a user who has ingested nothing.
        """
        ...

    async def delete_expired(self, cutoff_epoch: float) -> None:
        """Remove every record whose ``created_at`` is older than ``cutoff_epoch``."""
        ...

    async def count(self) -> int:
        ...

    async def reset(self) -> None:
        """Remove all stored vectors (used between ingest runs / in tests)."""
        ...
