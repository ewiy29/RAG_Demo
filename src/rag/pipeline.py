"""The RAG pipeline: the one object the API and CLI both drive.

It wires together a provider pair (embeddings + LLM) and a vector store, and
exposes two operations: ``ingest`` and ``ask``. Every ``ask`` emits a single
structured log record with the retrieved chunk ids and scores, end-to-end
latency, and token usage, which is the app's core observability surface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Settings, get_settings
from .conversation import ConversationStore, Message, build_conversation_store
from .generate import Answer, Citation, generate_answer
from .ingest import IngestResult, ingest_paths, ingest_uploads
from .logging_utils import get_logger
from .providers import build_providers
from .providers.base import ChatMessage, ChatUsage, EmbeddingProvider, LLMProvider
from .retrieve import RetrievedChunk, retrieve
from .vectorstore import build_store
from .vectorstore.base import SourceInfo, VectorStore

logger = get_logger("rag.pipeline")


@dataclass(frozen=True)
class AskResponse:
    query: str
    answer: str
    grounded: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    usage: ChatUsage = field(default_factory=ChatUsage)
    latency_ms: float = 0.0
    # The tenant this answer was scoped to (echoed back so a client that had one
    # minted for it can reuse the same id on later requests).
    user_id: str = ""
    # The conversation this answer belongs to (empty for the stateless /ask
    # path; populated for multi-turn /chat).
    conversation_id: str = ""


def _to_chat_history(messages: list[Message]) -> list[ChatMessage]:
    """Map stored conversation messages onto the provider-neutral history type."""

    return [ChatMessage(role=m.role, content=m.content) for m in messages]


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        store: VectorStore,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = store
        # Optional so stateless (/ask) tests can construct a pipeline without a
        # conversation backend; /chat requires it.
        self.conversation_store = conversation_store

    async def ingest(
        self, paths: list[str] | list[Path], *, user_id: str
    ) -> IngestResult:
        """Ingest files/directories from disk (corpus folders + tests)."""

        return await ingest_paths(
            paths, self.store, self.embedder, self.settings, user_id=user_id
        )

    async def ingest_uploads(
        self, uploads: list[tuple[str, bytes]], *, user_id: str
    ) -> IngestResult:
        """Ingest in-memory uploads (``(filename, data)``) for one user.

        The no-persistence path used by ``/ingest``: uploaded bytes are
        chunked, embedded, and stored straight from memory -- the raw upload is
        never staged on disk.
        """

        return await ingest_uploads(
            uploads, self.store, self.embedder, self.settings, user_id=user_id
        )

    async def ask(
        self,
        query: str,
        *,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> AskResponse:
        # ``conversation_id`` is a WS7 scaffold: it is accepted and logged for
        # correlation now, but multi-turn history/rewriting is not yet wired.
        start = time.perf_counter()

        # Log any failure with the timings gathered so far, then re-raise so the
        # API's typed-error envelope still forms. Without this a retrieve/
        # generate exception would propagate with no log at all.
        try:
            retrieved = await retrieve(
                query,
                self.store,
                self.embedder,
                k=self.settings.top_k,
                min_score=self.settings.min_score,
                user_id=user_id,
            )
            retrieval_ms = (time.perf_counter() - start) * 1000.0

            generate_start = time.perf_counter()
            answer: Answer = await generate_answer(query, retrieved, self.llm)
            generation_ms = (time.perf_counter() - generate_start) * 1000.0
        except Exception:
            logger.exception(
                "ask_failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                },
            )
            raise

        latency_ms = retrieval_ms + generation_ms

        # Defensive: WS2 guarantees a grounded answer carries >=1 verified
        # citation. Surface a violation loudly rather than silently trusting it.
        if answer.grounded and not answer.citations:
            logger.warning(
                "grounded_without_citation",
                extra={"user_id": user_id, "conversation_id": conversation_id},
            )

        logger.info(
            "ask",
            extra={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "grounded": answer.grounded,
                "latency_ms": round(latency_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "top_k": self.settings.top_k,
                "min_score": self.settings.min_score,
                "retrieved": [
                    {"id": c.id, "score": round(c.score, 4)} for c in retrieved
                ],
                "num_citations": len(answer.citations),
                "finish_reason": answer.finish_reason,
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
            },
        )

        return AskResponse(
            query=query,
            answer=answer.text,
            grounded=answer.grounded,
            citations=answer.citations,
            retrieved=retrieved,
            usage=answer.usage,
            latency_ms=latency_ms,
            user_id=user_id,
        )

    async def chat(
        self,
        query: str,
        *,
        user_id: str,
        conversation_id: str,
    ) -> AskResponse:
        """Answer a query in the context of an ongoing conversation.

        Reads the recent history, condenses the follow-up into a standalone
        query before retrieval, generates a grounded answer with the history in
        context, then persists both the user turn and the assistant reply. The
        grounding guarantee is unchanged: the answer must still be supported by
        (and cite) the retrieved corpus chunks.
        """

        if self.conversation_store is None:  # pragma: no cover - misconfiguration
            raise RuntimeError("chat() requires a conversation_store")

        start = time.perf_counter()

        stored_history = await self.conversation_store.get_history(
            user_id, conversation_id, limit=self.settings.conversation_history_limit
        )
        history = _to_chat_history(stored_history)

        # Log any failure with the timings gathered so far, then re-raise so the
        # API envelope still forms and the failure is not invisible.
        try:
            retrieved = await retrieve(
                query,
                self.store,
                self.embedder,
                k=self.settings.top_k,
                min_score=self.settings.min_score,
                user_id=user_id,
                history=history,
                llm=self.llm,
            )
            retrieval_ms = (time.perf_counter() - start) * 1000.0

            generate_start = time.perf_counter()
            answer: Answer = await generate_answer(
                query, retrieved, self.llm, history=history
            )
            generation_ms = (time.perf_counter() - generate_start) * 1000.0
        except Exception:
            logger.exception(
                "chat_failed",
                extra={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "history_len": len(history),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                },
            )
            raise

        # Persist the turn (write-through to durable + hot layer). Store the raw
        # user query (not the condensed form) so history reads back naturally.
        now = time.time()
        await self.conversation_store.append_message(
            user_id, conversation_id, Message(role="user", content=query, created_at=now)
        )
        await self.conversation_store.append_message(
            user_id,
            conversation_id,
            Message(role="assistant", content=answer.text, created_at=time.time()),
        )

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Defensive: a grounded answer must carry >=1 verified citation (WS2).
        if answer.grounded and not answer.citations:
            logger.warning(
                "grounded_without_citation",
                extra={"user_id": user_id, "conversation_id": conversation_id},
            )

        logger.info(
            "chat",
            extra={
                "user_id": user_id,
                "conversation_id": conversation_id,
                "history_len": len(history),
                "grounded": answer.grounded,
                "latency_ms": round(latency_ms, 2),
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "top_k": self.settings.top_k,
                "min_score": self.settings.min_score,
                "retrieved": [
                    {"id": c.id, "score": round(c.score, 4)} for c in retrieved
                ],
                "num_citations": len(answer.citations),
                "finish_reason": answer.finish_reason,
                "prompt_tokens": answer.usage.prompt_tokens,
                "completion_tokens": answer.usage.completion_tokens,
            },
        )

        return AskResponse(
            query=query,
            answer=answer.text,
            grounded=answer.grounded,
            citations=answer.citations,
            retrieved=retrieved,
            usage=answer.usage,
            latency_ms=latency_ms,
            user_id=user_id,
            conversation_id=conversation_id,
        )

    async def get_history(
        self, user_id: str, conversation_id: str
    ) -> list[Message]:
        """Return the recent messages of a conversation (for the UI)."""

        if self.conversation_store is None:  # pragma: no cover - misconfiguration
            raise RuntimeError("get_history() requires a conversation_store")
        return await self.conversation_store.get_history(
            user_id, conversation_id, limit=self.settings.conversation_history_limit
        )

    async def list_conversations(self, user_id: str) -> list[str]:
        """Return the ids of a user's conversations (for the UI)."""

        if self.conversation_store is None:  # pragma: no cover - misconfiguration
            raise RuntimeError("list_conversations() requires a conversation_store")
        return await self.conversation_store.list_conversations(user_id)

    async def list_sources(self, user_id: str) -> list[SourceInfo]:
        """Return the distinct sources a user has ingested (for the UI)."""

        return await self.store.list_sources(user_id)

    async def delete_source(self, source: str, *, user_id: str) -> None:
        """Delete a single ingested source for one user (per-file removal)."""

        await self.store.delete_by_source(source, user_id=user_id)

    async def purge_user(self, user_id: str) -> None:
        """Delete all of a user's stored data ("delete all my data")."""

        await self.store.delete_by_user(user_id)

    async def purge_conversations(self, user_id: str) -> None:
        """Delete all of a user's conversation history."""

        if self.conversation_store is None:  # pragma: no cover - misconfiguration
            raise RuntimeError("purge_conversations() requires a conversation_store")
        await self.conversation_store.delete_by_user(user_id)

    async def cleanup_expired(self, ttl_seconds: Optional[int] = None) -> None:
        """Remove per-user data older than the TTL (on-demand, no scheduler).

        ``ttl_seconds`` defaults to ``settings.session_ttl_seconds``. This is
        the lightweight cleanup hook for the ephemeral per-user model; a
        production deployment would drive it from a scheduled job.
        """

        ttl = ttl_seconds if ttl_seconds is not None else self.settings.session_ttl_seconds
        await self.store.delete_expired(time.time() - ttl)


def build_pipeline(
    settings: Settings | None = None,
    store_kind: str = "chroma",
) -> RagPipeline:
    """Construct a pipeline from settings (providers + store chosen for you)."""

    settings = settings or get_settings()
    providers = build_providers(settings)
    store = build_store(settings, kind=store_kind)
    conversation_store = build_conversation_store(settings)
    return RagPipeline(
        settings=settings,
        embedder=providers.embedder,
        llm=providers.llm,
        store=store,
        conversation_store=conversation_store,
    )
