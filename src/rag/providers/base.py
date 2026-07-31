"""Provider-agnostic interfaces for embeddings and chat completion.

The rest of the app depends only on these Protocols, never on a concrete SDK.
This is what lets us default to OpenAI in production while running the entire
test suite against a deterministic fake with no network and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChatUsage:
    """Token accounting for a single chat call (used for observability)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatResult:
    """The model's reply plus token usage."""

    text: str
    usage: ChatUsage


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Must be deterministic for a given input."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input string, in order."""
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a chat completion from a system + user message."""

    def chat(self, system: str, user: str) -> ChatResult:
        ...
