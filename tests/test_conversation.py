"""Tests for WS7 conversation state + the multi-turn /chat surface.

Covers the in-memory test double, the shipped two-tier ``RedisConversationStore``
over ``fakeredis`` (append / last-N read / native TTL / rehydrate-from-durable on
a hot-layer miss / list / delete), and an end-to-end /chat smoke test showing
prior turns are condensed into the query so a bare follow-up still retrieves.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from doubles import InMemoryConversationStore, build_fake_pipeline
from rag.api import create_app
from rag.config import Settings
from rag.conversation import Message
from rag.conversation.redis_store import RedisConversationStore
from rag.conversation.sqlite_durable import SqliteDurableBackend

FIXTURES = Path(__file__).parent / "fixtures"
USER = "conv-user"


def _msg(role: str, content: str, ts: float = 1.0) -> Message:
    return Message(role=role, content=content, created_at=ts)


# --- in-memory test double ---------------------------------------------------


async def test_in_memory_double_append_get_last_n_and_list():
    store = InMemoryConversationStore()
    for i in range(5):
        await store.append_message(USER, "c1", _msg("user", f"m{i}", ts=float(i)))

    last_two = await store.get_history(USER, "c1", limit=2)
    assert [m.content for m in last_two] == ["m3", "m4"]

    assert await store.list_conversations(USER) == ["c1"]


async def test_in_memory_double_delete_by_user_isolated():
    store = InMemoryConversationStore()
    await store.append_message(USER, "c1", _msg("user", "hi"))
    await store.append_message("other", "c9", _msg("user", "hello"))

    await store.delete_by_user(USER)

    assert await store.get_history(USER, "c1", limit=10) == []
    assert await store.list_conversations(USER) == []
    # Another tenant's conversation is untouched.
    assert await store.list_conversations("other") == ["c9"]


# --- two-tier Redis (fakeredis) + SQLite durable -----------------------------


def _redis_store(tmp_path, *, ttl_seconds: int = 100) -> RedisConversationStore:
    durable = SqliteDurableBackend(path=str(tmp_path / "conv.sqlite3"))
    # redis_url unset -> in-process fakeredis (no server needed).
    return RedisConversationStore(durable, redis_url="", ttl_seconds=ttl_seconds)


async def test_redis_store_append_read_last_n_and_sets_ttl(tmp_path):
    store = _redis_store(tmp_path, ttl_seconds=100)
    for i in range(4):
        await store.append_message(USER, "c1", _msg("user", f"m{i}", ts=float(i)))

    last_two = await store.get_history(USER, "c1", limit=2)
    assert [m.content for m in last_two] == ["m2", "m3"]

    # Native per-key TTL was set (and refreshed) on append.
    client = await store._ensure_client()
    ttl = await client.ttl("conv:conv-user:c1")
    assert 0 < ttl <= 100


async def test_redis_store_rehydrates_from_durable_on_hot_miss(tmp_path):
    # Instance A writes through to the shared durable file + its own hot layer.
    writer = _redis_store(tmp_path)
    await writer.append_message(USER, "c1", _msg("user", "first"))
    await writer.append_message(USER, "c1", _msg("assistant", "second"))

    # Instance B shares the durable file but has a fresh (empty) hot layer,
    # simulating a Redis flush/restart. Its read must rehydrate from durable.
    reader = _redis_store(tmp_path)
    reader_client = await reader._ensure_client()
    assert await reader_client.exists("conv:conv-user:c1") == 0  # cold hot-layer

    history = await reader.get_history(USER, "c1", limit=10)
    assert [(m.role, m.content) for m in history] == [
        ("user", "first"),
        ("assistant", "second"),
    ]
    # After rehydration the hot layer is warm again.
    assert await reader_client.exists("conv:conv-user:c1") == 1


async def test_redis_store_list_and_delete_by_user(tmp_path):
    store = _redis_store(tmp_path)
    await store.append_message(USER, "c1", _msg("user", "a"))
    await store.append_message(USER, "c2", _msg("user", "b"))
    await store.append_message("other", "c3", _msg("user", "c"))

    assert set(await store.list_conversations(USER)) == {"c1", "c2"}

    await store.delete_by_user(USER)
    assert await store.list_conversations(USER) == []
    # Hot-layer keys were evicted too.
    client = await store._ensure_client()
    assert await client.exists("conv:conv-user:c1") == 0
    # Another tenant is untouched.
    assert await store.list_conversations("other") == ["c3"]


# --- end-to-end /chat multi-turn --------------------------------------------


def _async_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_chat_multi_turn_condenses_history_and_persists(tmp_path):
    settings = Settings(top_k=4, min_score=0.2)
    # build_fake_pipeline wires the in-memory conversation double directly.
    pipe = build_fake_pipeline(settings)
    await pipe.ingest([str(FIXTURES / "water.md")], user_id=USER)

    headers = {"X-User-Id": USER, "X-Conversation-Id": "chat-1"}

    async with _async_client(create_app(pipeline=pipe)) as client:
        first = await client.post(
            "/chat", json={"query": "What is water composed of?"}, headers=headers
        )
        assert first.status_code == 200
        body1 = first.json()
        assert body1["grounded"] is True
        assert body1["conversation_id"] == "chat-1"
        assert first.headers["X-Conversation-Id"] == "chat-1"
        assert any("water.md" in c["source"] for c in body1["citations"])

        # A bare follow-up: on its own it would not retrieve water, but folding
        # the prior turn into the query (condensation) makes it grounded.
        second = await client.post(
            "/chat", json={"query": "Tell me more about it."}, headers=headers
        )
        body2 = second.json()
        assert body2["grounded"] is True
        assert any("water.md" in c["source"] for c in body2["citations"])

        # Both turns (user + assistant each) were persisted.
        history = await client.get("/conversations/chat-1", headers=headers)
        messages = history.json()["messages"]
        assert [m["role"] for m in messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert messages[0]["content"] == "What is water composed of?"

        # The conversation is listed for the user.
        listing = await client.get("/conversations", headers=headers)
        assert listing.json()["conversations"] == ["chat-1"]

        # Purge clears it.
        await client.delete("/conversations", headers=headers)
        assert (await client.get("/conversations", headers=headers)).json()[
            "conversations"
        ] == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
