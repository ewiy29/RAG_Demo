"""In-memory conversation store -- a TEST DOUBLE ONLY, never shipped.

Relocated under ``tests/`` (WS10). This exists so the offline test suite can
exercise the multi-turn history/query-rewriting flow without a Redis server or a
SQLite file. It is deliberately NOT the shipped persistence: an in-process dict
is lost on restart and is not shared across replicas -- the exact reasons the
project rejected in-memory for both the vector store and conversation state. The
shipped store is ``RedisConversationStore`` (Redis hot + durable). Mirrors the
in-memory vector store's "reference/test double, not production" role.

The methods are ``async`` with synchronous bodies, so each runs atomically on
the single-threaded event loop (the same concurrency-safety argument as the
in-memory vector store).
"""

from __future__ import annotations

from rag.conversation.base import Message, conversation_key


class InMemoryConversationStore:
    """Dict-backed conversation history for tests only (no TTL, no persistence)."""

    def __init__(self) -> None:
        # key -> ordered list of messages (append order == chronological order).
        self._conversations: dict[str, list[Message]] = {}
        # user_id -> conversation ids in first-seen order (for list_conversations).
        self._by_user: dict[str, list[str]] = {}

    async def append_message(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        key = conversation_key(user_id, conversation_id)
        self._conversations.setdefault(key, []).append(message)
        seen = self._by_user.setdefault(user_id, [])
        if conversation_id not in seen:
            seen.append(conversation_id)

    async def get_history(
        self, user_id: str, conversation_id: str, limit: int
    ) -> list[Message]:
        if limit <= 0:
            return []
        key = conversation_key(user_id, conversation_id)
        return list(self._conversations.get(key, []))[-limit:]

    async def list_conversations(self, user_id: str) -> list[str]:
        return list(self._by_user.get(user_id, []))

    async def delete_by_user(self, user_id: str) -> None:
        for conversation_id in self._by_user.pop(user_id, []):
            self._conversations.pop(conversation_key(user_id, conversation_id), None)
