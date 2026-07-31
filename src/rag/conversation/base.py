"""Conversation store interface + data model (multi-turn chat state).

Conversation state is a **KV / session** pattern, deliberately NOT relational
and NOT document: the only access shapes are *append a message*, *read the last
N messages for a conversation*, *list a user's conversations*, and *expire the
whole conversation on TTL*. There are no joins and no rich document queries, so
the industry-standard fit is a hot **key-value** layer (Redis) fronting a
durable persistence layer -- not a query-oriented database.

Two tiers (see ``redis_store.py`` / ``sqlite_durable.py``)
---------------------------------------------------------
* **Hot / speed layer = Redis** (``redis.asyncio``; ``fakeredis`` in-proc when no
  server is configured). One key per ``(user_id, conversation_id)`` holds a LIST
  of serialized message blobs, with a native per-key ``EXPIRE`` = the session
  TTL. TTL is native to the hot layer -- there is no hand-rolled sweep.
* **Durable / persistence layer** = source of truth for recovery, so a Redis
  flush/restart does not lose in-TTL conversations. The demo ships embedded
  SQLite; production swaps in Postgres/Cosmos behind the same
  ``DurableConversationBackend`` interface.

Reads are Redis-first and rehydrate from the durable tier on a miss; writes go
write-through to durable.

Contract every backend must honour
----------------------------------
* **Async.** Every method is a coroutine so a store backed by network I/O never
  blocks the event loop (mirrors the ``VectorStore`` contract).
* **Tenancy.** Keys are scoped ``conv:{user_id}:{conversation_id}`` (the WS6
  ``user_id`` GUID), so one tenant can never read another's conversation.
* **Native TTL.** Expiry is a per-key property of the hot layer, refreshed on
  every append -- an idle conversation disappears on its own.

The in-memory implementation (``tests/doubles/in_memory_conversation.py``) is a
**test double only** and is never the shipped persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Message:
    """One turn in a conversation.

    ``role`` is ``"user"`` or ``"assistant"`` (the two roles this app persists);
    ``content`` is the raw text; ``created_at`` is epoch seconds. Messages are
    serialized to a small JSON blob to cross the Redis/SQLite boundary.
    """

    role: str
    content: str
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            created_at=float(data.get("created_at", 0.0)),
        )


def conversation_key(user_id: str, conversation_id: str) -> str:
    """The hot-layer key for one conversation: ``conv:{user_id}:{conversation_id}``."""

    return f"conv:{user_id}:{conversation_id}"


@runtime_checkable
class ConversationStore(Protocol):
    """Async multi-turn conversation state (Redis hot + durable, or a test double)."""

    async def append_message(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        """Append ``message`` to the conversation and refresh its TTL."""
        ...

    async def get_history(
        self, user_id: str, conversation_id: str, limit: int
    ) -> list[Message]:
        """Return up to the last ``limit`` messages, oldest-first.

        Reads the hot layer first and rehydrates from the durable tier on a miss
        (so a conversation survives a Redis flush/restart within its TTL).
        """
        ...

    async def list_conversations(self, user_id: str) -> list[str]:
        """Return the ids of every conversation belonging to ``user_id``."""
        ...

    async def delete_by_user(self, user_id: str) -> None:
        """Delete all of a user's conversations ("delete all my data")."""
        ...


@runtime_checkable
class DurableConversationBackend(Protocol):
    """The persistence tier behind the Redis hot layer.

    Embedded SQLite in the demo; Postgres/Cosmos in production. It is the source
    of truth used to rehydrate the hot layer and to enumerate a user's
    conversations (the hot layer cannot list TTL-expired keys reliably).
    """

    async def append(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        ...

    async def get_history(
        self, user_id: str, conversation_id: str, limit: int | None = None
    ) -> list[Message]:
        ...

    async def list_conversations(self, user_id: str) -> list[str]:
        ...

    async def delete_by_user(self, user_id: str) -> None:
        ...
