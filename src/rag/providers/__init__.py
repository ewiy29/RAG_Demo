"""Provider selection.

``build_providers`` returns an ``(embedding, llm)`` pair chosen from settings,
so callers never branch on the provider name themselves.
"""

from __future__ import annotations

from ..config import Settings
from .base import ChatResult, ChatUsage, EmbeddingProvider, LLMProvider

__all__ = [
    "ChatResult",
    "ChatUsage",
    "EmbeddingProvider",
    "LLMProvider",
    "build_providers",
]


def build_providers(settings: Settings) -> tuple[EmbeddingProvider, LLMProvider]:
    provider = settings.provider.lower()

    if provider == "fake":
        from .fake_provider import FakeEmbeddingProvider, FakeLLMProvider

        return FakeEmbeddingProvider(), FakeLLMProvider()

    if provider == "openai":
        from .openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider

        embedder = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key, model=settings.embedding_model
        )
        llm = OpenAILLMProvider(
            api_key=settings.openai_api_key, model=settings.chat_model
        )
        return embedder, llm

    raise ValueError(
        f"Unknown provider {settings.provider!r}. Use 'openai' or 'fake'."
    )
