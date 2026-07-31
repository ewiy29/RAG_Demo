"""Conversation store selection (multi-turn chat state).

``build_conversation_store`` returns the shipped two-tier store (Redis hot +
SQLite durable). The in-memory store is a test double only and is never the
production path; it lives under ``tests/doubles`` (WS10) and tests inject
``InMemoryConversationStore`` directly rather than selecting it via config.
"""

from __future__ import annotations

from ..config import Settings
from .base import (
    ConversationStore,
    DurableConversationBackend,
    Message,
    conversation_key,
)

__all__ = [
    "ConversationStore",
    "DurableConversationBackend",
    "Message",
    "conversation_key",
    "build_conversation_store",
]


def _build_durable_backend(settings: Settings) -> DurableConversationBackend:
    if settings.conversation_durable_backend == "sqlite":
        from .sqlite_durable import SqliteDurableBackend

        return SqliteDurableBackend(path=settings.conversation_durable_path)
    raise ValueError(
        f"Unknown durable conversation backend "
        f"{settings.conversation_durable_backend!r}. Use 'sqlite'."
    )


def build_conversation_store(settings: Settings) -> ConversationStore:
    """Construct the conversation store selected by ``settings``."""

    kind = settings.conversation_store.lower()
    if kind == "redis":
        from .redis_store import RedisConversationStore

        return RedisConversationStore(
            _build_durable_backend(settings),
            redis_url=settings.redis_url,
            ttl_seconds=settings.session_ttl_seconds,
        )
    raise ValueError(
        f"Unknown conversation store kind {kind!r}. Use 'redis'."
    )
