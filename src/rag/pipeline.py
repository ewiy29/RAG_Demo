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

from .config import Settings, get_settings
from .generate import Answer, Citation, generate_answer
from .ingest import IngestResult, ingest_paths
from .logging_utils import get_logger
from .providers import build_providers
from .providers.base import ChatUsage, EmbeddingProvider, LLMProvider
from .retrieve import RetrievedChunk, retrieve
from .vectorstore import build_store
from .vectorstore.base import VectorStore

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


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = store

    def ingest(self, paths: list[str] | list[Path]) -> IngestResult:
        return ingest_paths(paths, self.store, self.embedder, self.settings)

    def ask(self, query: str) -> AskResponse:
        start = time.perf_counter()

        retrieved = retrieve(
            query,
            self.store,
            self.embedder,
            k=self.settings.top_k,
            min_score=self.settings.min_score,
        )
        answer: Answer = generate_answer(query, retrieved, self.llm)

        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "ask",
            extra={
                "grounded": answer.grounded,
                "latency_ms": round(latency_ms, 2),
                "top_k": self.settings.top_k,
                "min_score": self.settings.min_score,
                "retrieved": [
                    {"id": c.id, "score": round(c.score, 4)} for c in retrieved
                ],
                "num_citations": len(answer.citations),
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
        )


def build_pipeline(
    settings: Settings | None = None,
    store_kind: str = "chroma",
) -> RagPipeline:
    """Construct a pipeline from settings (providers + store chosen for you)."""

    settings = settings or get_settings()
    embedder, llm = build_providers(settings)
    store = build_store(settings, kind=store_kind)
    return RagPipeline(settings=settings, embedder=embedder, llm=llm, store=store)
