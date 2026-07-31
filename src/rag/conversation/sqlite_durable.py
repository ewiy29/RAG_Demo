"""Embedded SQLite durable backend for conversation history.

This is the demo's persistence tier: a zero-server, stdlib ``sqlite3`` file that
is the source of truth so conversations survive a Redis flush/restart within
their TTL, and that can enumerate a user's conversations. Production swaps this
for Postgres/Cosmos behind the same ``DurableConversationBackend`` interface --
only the connection string changes.

``sqlite3`` is blocking, so every call is offloaded to a threadpool via
``asyncio.to_thread`` (the same pattern the Chroma vector store uses), keeping
the async event loop responsive. A fresh short-lived connection per operation
keeps this thread-safe without a shared-connection lock -- fine at demo scale.
"""

from __future__ import annotations

import asyncio
import sqlite3

from .base import Message

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    created_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv
    ON messages (user_id, conversation_id, id);
"""


class SqliteDurableBackend:
    """Durable conversation persistence backed by a local SQLite file."""

    def __init__(self, path: str = "conversations.sqlite3") -> None:
        self._path = path
        # Schema is created lazily on first use so merely constructing a pipeline
        # (which happens for the stateless /ask path too) never touches disk.
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        # ``check_same_thread=False`` because each call may run on a different
        # threadpool worker; we still use one connection per operation.
        conn = sqlite3.connect(self._path, check_same_thread=False)
        if not self._schema_ready:
            # ``CREATE ... IF NOT EXISTS`` is idempotent, so a first-use race
            # between threadpool workers is harmless.
            conn.executescript(_SCHEMA)
            conn.commit()
            self._schema_ready = True
        return conn

    def _append_sync(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO messages "
                "(user_id, conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    conversation_id,
                    message.role,
                    message.content,
                    message.created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _get_history_sync(
        self, user_id: str, conversation_id: str, limit: int | None
    ) -> list[Message]:
        conn = self._connect()
        try:
            if limit is None:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE user_id = ? AND conversation_id = ? ORDER BY id ASC",
                    (user_id, conversation_id),
                ).fetchall()
            else:
                # Take the most recent ``limit`` rows, then restore oldest-first.
                rows = conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE user_id = ? AND conversation_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (user_id, conversation_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
        finally:
            conn.close()
        return [
            Message(role=role, content=content, created_at=created_at)
            for role, content, created_at in rows
        ]

    def _list_conversations_sync(self, user_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT conversation_id FROM messages WHERE user_id = ? "
                "GROUP BY conversation_id ORDER BY MAX(id) DESC",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()
        return [row[0] for row in rows]

    def _delete_by_user_sync(self, user_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
            conn.commit()
        finally:
            conn.close()

    async def append(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        await asyncio.to_thread(
            self._append_sync, user_id, conversation_id, message
        )

    async def get_history(
        self, user_id: str, conversation_id: str, limit: int | None = None
    ) -> list[Message]:
        return await asyncio.to_thread(
            self._get_history_sync, user_id, conversation_id, limit
        )

    async def list_conversations(self, user_id: str) -> list[str]:
        return await asyncio.to_thread(self._list_conversations_sync, user_id)

    async def delete_by_user(self, user_id: str) -> None:
        await asyncio.to_thread(self._delete_by_user_sync, user_id)
