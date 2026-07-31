"""Test doubles relocated out of the shipped package (WS10).

The fake provider and the in-memory vector/conversation stores are test aids,
not production code, so they live here under ``tests/`` rather than in
``src/rag``. The production factories only build the real backends (OpenAI +
Chroma + Redis); tests inject these doubles directly into a ``RagPipeline``.

``build_fake_pipeline`` is the offline equivalent of ``rag.build_pipeline``: it
wires the fakes together so integration tests keep a one-line pipeline setup
without going through the (now production-only) factories.
"""

from __future__ import annotations

from rag.config import Settings
from rag.pipeline import RagPipeline

from .fake_provider import FakeEmbeddingProvider, FakeLLMProvider
from .in_memory_conversation import InMemoryConversationStore
from .in_memory_store import InMemoryVectorStore

__all__ = [
    "FakeEmbeddingProvider",
    "FakeLLMProvider",
    "InMemoryConversationStore",
    "InMemoryVectorStore",
    "build_fake_pipeline",
]


def build_fake_pipeline(settings: Settings) -> RagPipeline:
    """Construct a fully offline pipeline backed by the test doubles.

    Mirrors what ``rag.build_pipeline`` used to do when pointed at the fake
    provider + in-memory stores, but wires the doubles directly (dependency
    injection) rather than selecting them through the production factories.
    """

    return RagPipeline(
        settings=settings,
        embedder=FakeEmbeddingProvider(),
        llm=FakeLLMProvider(),
        store=InMemoryVectorStore(),
        conversation_store=InMemoryConversationStore(),
    )
