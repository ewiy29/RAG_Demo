"""Two-tier conversation store: Redis hot layer + a durable backend.

This is the **shipped** conversation store. The hot layer is Redis
(``redis.asyncio``) holding one LIST per ``conv:{user_id}:{conversation_id}``
key with a native per-key ``EXPIRE`` = the session TTL; the durable backend
(embedded SQLite by default) is the source of truth for recovery and
enumeration.

Client selection ("same redis-py API either way"):
* ``redis_url`` set + reachable  -> real ``redis.asyncio`` client.
* ``redis_url`` unset, or set but unreachable -> in-process ``fakeredis`` so a
  clone-and-run demo works with no Redis server installed.

The choice is made lazily on first use (``_ensure_client``) because building the
pipeline is synchronous while pinging Redis is async; falling back on a
connection error keeps the no-install path working even if a stale ``redis_url``
is configured.

Read path is Redis-first and **rehydrates from durable on a miss** (a flushed or
restarted Redis repopulates from SQLite for any conversation still within TTL).
Write path is **write-through**: every append writes both tiers (write-behind is
a later optimization; write-through keeps the demo simple and loss-free).
"""

from __future__ import annotations

import json

from ..logging_utils import get_logger
from .base import DurableConversationBackend, Message, conversation_key

logger = get_logger("rag.conversation.redis")

_SCAN_COUNT = 100


class RedisConversationStore:
    """Redis hot layer in front of a durable persistence backend."""

    def __init__(
        self,
        durable: DurableConversationBackend,
        *,
        redis_url: str = "",
        ttl_seconds: int = 86400,
    ) -> None:
        self._durable = durable
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._client = None  # resolved lazily on first use

    async def _ensure_client(self):
        """Return the Redis client, choosing real vs. fakeredis on first use."""

        if self._client is not None:
            return self._client

        if self._redis_url:
            try:
                import redis.asyncio as redis

                client = redis.from_url(
                    self._redis_url, encoding="utf-8", decode_responses=True
                )
                await client.ping()
                self._client = client
                return self._client
            except Exception as exc:
                # Unreachable/misconfigured Redis: fall back to the in-process
                # fake so clone-and-run still works (same redis-py API).
                logger.warning("redis unavailable (%s); falling back to fakeredis", exc)

        from fakeredis import aioredis as fakeredis_aioredis

        self._client = fakeredis_aioredis.FakeRedis(decode_responses=True)
        return self._client

    async def append_message(
        self, user_id: str, conversation_id: str, message: Message
    ) -> None:
        # Write-through: durable first (source of truth), then the hot layer.
        await self._durable.append(user_id, conversation_id, message)

        client = await self._ensure_client()
        key = conversation_key(user_id, conversation_id)
        await client.rpush(key, json.dumps(message.to_dict()))
        # Native TTL: refresh the whole conversation's expiry on every append.
        await client.expire(key, self._ttl)

    async def get_history(
        self, user_id: str, conversation_id: str, limit: int
    ) -> list[Message]:
        if limit <= 0:
            return []

        client = await self._ensure_client()
        key = conversation_key(user_id, conversation_id)

        if await client.exists(key):
            raw = await client.lrange(key, -limit, -1)
            return [Message.from_dict(json.loads(blob)) for blob in raw]

        # Hot-layer miss: rehydrate from durable, repopulate Redis (with TTL),
        # and return the last-N.
        durable_history = await self._durable.get_history(user_id, conversation_id)
        if not durable_history:
            return []

        blobs = [json.dumps(m.to_dict()) for m in durable_history]
        await client.rpush(key, *blobs)
        await client.expire(key, self._ttl)
        return durable_history[-limit:]

    async def list_conversations(self, user_id: str) -> list[str]:
        # Enumeration comes from durable: the hot layer's TTL'd keys are not a
        # reliable listing (an expired-but-still-durable conversation would be
        # missed, and SCAN over a large keyspace is costly).
        return await self._durable.list_conversations(user_id)

    async def delete_by_user(self, user_id: str) -> None:
        # Clear the durable tier, then evict any live hot-layer keys.
        await self._durable.delete_by_user(user_id)

        client = await self._ensure_client()
        pattern = conversation_key(user_id, "*")
        stale_keys = [key async for key in client.scan_iter(match=pattern, count=_SCAN_COUNT)]
        if stale_keys:
            await client.delete(*stale_keys)
