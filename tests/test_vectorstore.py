"""Contract tests for the vector store data model (WS5).

Exercised against ``InMemoryVectorStore`` as the reference implementation of the
store contract: upsert-by-id, consistent embedding dimensionality,
delete/delete-by-source, metadata filtering on query, and typed
``ChunkMetadata`` on hits. The Chroma adapter is expected to honour the same
contract but is not run here (no external dependency in the offline suite).
"""

from __future__ import annotations

import pytest

from doubles import InMemoryVectorStore
from rag.vectorstore import ChunkMetadata, ChunkRecord
from rag.vectorstore.base import QueryHit


def _record(
    id_: str,
    embedding,
    source: str,
    chunk_index: int,
    text: str = "",
    *,
    user_id: str = "",
    created_at: float = 0.0,
) -> ChunkRecord:
    return ChunkRecord(
        id=id_,
        embedding=embedding,
        text=text or id_,
        metadata=ChunkMetadata(
            source=source,
            chunk_index=chunk_index,
            user_id=user_id,
            created_at=created_at,
        ),
    )


async def test_add_is_upsert_by_id_no_duplicate():
    store = InMemoryVectorStore()
    await store.add([_record("doc.md#0", [1.0, 0.0], "doc.md", 0, text="old")])
    # Re-adding the same id replaces in place rather than duplicating.
    await store.add([_record("doc.md#0", [0.0, 1.0], "doc.md", 0, text="new")])

    assert await store.count() == 1
    hits = await store.query([0.0, 1.0], k=5)
    assert len(hits) == 1
    assert hits[0].text == "new"


async def test_query_returns_typed_chunk_metadata():
    store = InMemoryVectorStore()
    await store.add([_record("doc.md#3", [1.0, 0.0], "doc.md", 3, text="body")])

    hit = (await store.query([1.0, 0.0], k=1))[0]
    assert isinstance(hit, QueryHit)
    assert isinstance(hit.metadata, ChunkMetadata)
    assert hit.metadata.source == "doc.md"
    assert hit.metadata.chunk_index == 3


async def test_delete_removes_by_id():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("a#0", [1.0, 0.0], "a", 0),
            _record("b#0", [0.0, 1.0], "b", 0),
        ]
    )
    await store.delete(["a#0"])

    assert await store.count() == 1
    remaining = {hit.id for hit in await store.query([1.0, 1.0], k=5)}
    assert remaining == {"b#0"}
    # Deleting an absent id is a no-op.
    await store.delete(["missing#0"])
    assert await store.count() == 1


async def test_delete_by_source_removes_all_chunks_of_that_source():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("a#0", [1.0, 0.0], "a", 0),
            _record("a#1", [0.9, 0.1], "a", 1),
            _record("b#0", [0.0, 1.0], "b", 0),
        ]
    )
    await store.delete_by_source("a")

    assert await store.count() == 1
    remaining = {hit.id for hit in await store.query([1.0, 1.0], k=5)}
    assert remaining == {"b#0"}


async def test_query_where_filters_by_metadata():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("a#0", [1.0, 0.0], "a", 0),
            _record("b#0", [1.0, 0.0], "b", 0),
        ]
    )
    # Identical vectors; the filter is what scopes the result.
    hits = await store.query([1.0, 0.0], k=5, where={"source": "b"})
    assert [hit.id for hit in hits] == ["b#0"]


async def test_dimension_mismatch_on_add_raises():
    store = InMemoryVectorStore()
    await store.add([_record("a#0", [1.0, 0.0, 0.0], "a", 0)])
    with pytest.raises(ValueError):
        await store.add([_record("b#0", [1.0, 0.0], "b", 0)])


async def test_dimension_mismatch_on_query_raises():
    store = InMemoryVectorStore()
    await store.add([_record("a#0", [1.0, 0.0, 0.0], "a", 0)])
    with pytest.raises(ValueError):
        await store.query([1.0, 0.0], k=1)


async def test_reset_clears_records_and_dimension():
    store = InMemoryVectorStore()
    await store.add([_record("a#0", [1.0, 0.0, 0.0], "a", 0)])
    await store.reset()

    assert await store.count() == 0
    # Dimension is forgotten, so a new (different-dim) corpus can be stored.
    await store.add([_record("b#0", [1.0, 0.0], "b", 0)])
    assert await store.count() == 1


async def test_query_where_user_id_isolates_tenants():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("ua/doc#0", [1.0, 0.0], "doc", 0, user_id="ua"),
            _record("ub/doc#0", [1.0, 0.0], "doc", 0, user_id="ub"),
        ]
    )
    # Identical vectors + same source; only the user_id filter scopes the hit.
    hits = await store.query([1.0, 0.0], k=5, where={"user_id": "ua"})
    assert [hit.id for hit in hits] == ["ua/doc#0"]
    assert hits[0].metadata.user_id == "ua"


async def test_delete_by_source_scoped_to_user_leaves_other_user():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("ua/notes#0", [1.0, 0.0], "notes", 0, user_id="ua"),
            _record("ub/notes#0", [0.0, 1.0], "notes", 0, user_id="ub"),
        ]
    )
    # Deleting user A's "notes" must not remove user B's same-named source.
    await store.delete_by_source("notes", user_id="ua")

    remaining = {hit.id for hit in await store.query([1.0, 1.0], k=5)}
    assert remaining == {"ub/notes#0"}


async def test_delete_by_user_removes_only_that_user():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("ua/a#0", [1.0, 0.0], "a", 0, user_id="ua"),
            _record("ua/b#0", [0.9, 0.1], "b", 0, user_id="ua"),
            _record("ub/a#0", [0.0, 1.0], "a", 0, user_id="ub"),
        ]
    )
    await store.delete_by_user("ua")

    remaining = {hit.id for hit in await store.query([1.0, 1.0], k=5)}
    assert remaining == {"ub/a#0"}


async def test_delete_expired_removes_old_records_only():
    store = InMemoryVectorStore()
    await store.add(
        [
            _record("old#0", [1.0, 0.0], "old", 0, user_id="ua", created_at=100.0),
            _record("new#0", [0.0, 1.0], "new", 0, user_id="ua", created_at=1000.0),
        ]
    )
    # Cutoff between the two timestamps expires only the older record.
    await store.delete_expired(500.0)

    remaining = {hit.id for hit in await store.query([1.0, 1.0], k=5)}
    assert remaining == {"new#0"}
