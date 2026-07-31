"""Provider selection.

``build_providers(settings)`` returns a :class:`Providers` bundle (embedding +
chat) chosen from settings, so callers never branch on the provider name
themselves and never risk swapping the positional pair.

Selection uses a small ``name -> builder`` **registry** rather than an
``if/elif`` chain: adding a vendor (Anthropic, a local model, ...) is a
``@register_provider("name")`` registration implementing the base protocols +
the typed ``ProviderError`` mapping, not an edit to the dispatch. Providers are
built **once** (at startup, via ``build_pipeline``) and reused, so a single
shared client is pooled across the process rather than rebuilt per request.

Only the real OpenAI provider ships. The deterministic offline *fake* provider
is a test aid and lives under ``tests/doubles`` (WS10); tests inject it directly
into the pipeline rather than selecting it via a config string.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..config import Settings
from .base import ChatResult, ChatUsage, EmbeddingProvider, LLMProvider

__all__ = [
    "ChatResult",
    "ChatUsage",
    "EmbeddingProvider",
    "LLMProvider",
    "Providers",
    "build_providers",
    "register_provider",
]


@dataclass(frozen=True)
class Providers:
    """The embedding + chat providers for a running pipeline.

    A named bundle instead of a positional ``(embedder, llm)`` tuple so callers
    are self-documenting and can't accidentally swap the two.
    """

    embedder: EmbeddingProvider
    llm: LLMProvider


# name -> builder. A builder takes validated Settings and returns a Providers
# bundle. Registration is the extension point for new vendors.
_PROVIDER_BUILDERS: dict[str, Callable[[Settings], Providers]] = {}


def register_provider(
    name: str,
) -> Callable[[Callable[[Settings], Providers]], Callable[[Settings], Providers]]:
    """Register a provider builder under ``name`` (case-insensitive)."""

    def decorator(
        builder: Callable[[Settings], Providers],
    ) -> Callable[[Settings], Providers]:
        _PROVIDER_BUILDERS[name.lower()] = builder
        return builder

    return decorator


@register_provider("openai")
def _build_openai(settings: Settings) -> Providers:
    from .openai_provider import (
        OpenAIEmbeddingProvider,
        OpenAILLMProvider,
        build_openai_client,
    )

    # One shared client (connection pooling) with the request timeout/retry
    # policy from config, injected into both providers.
    client = build_openai_client(
        settings.openai_api_key,
        timeout=settings.request_timeout_seconds,
        max_retries=settings.max_retries,
    )
    embedder = OpenAIEmbeddingProvider(
        client,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    llm = OpenAILLMProvider(
        client,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
    )
    return Providers(embedder=embedder, llm=llm)


def build_providers(settings: Settings) -> Providers:
    builder = _PROVIDER_BUILDERS.get(settings.provider.lower())
    if builder is None:
        known = ", ".join(sorted(_PROVIDER_BUILDERS)) or "(none registered)"
        raise ValueError(
            f"Unknown provider {settings.provider!r}. Registered: {known}."
        )
    return builder(settings)
