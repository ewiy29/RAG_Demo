"""Provider-agnostic interfaces for embeddings and chat completion.

The rest of the app depends only on these Protocols, never on a concrete SDK.
This is what lets us default to OpenAI in production while running the entire
test suite against a deterministic fake with no network and no API key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    """One prior turn passed to :meth:`LLMProvider.chat` as conversation history.

    ``role`` is ``"user"`` or ``"assistant"``. This is a provider-neutral shape
    (the provider layer must not depend on the conversation module), so callers
    map their own message type onto it.
    """

    role: str
    content: str


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
    """The model's reply plus token usage and why it stopped.

    ``finish_reason`` mirrors the provider's stop reason (e.g. ``"stop"`` for a
    complete reply, ``"length"`` when the output was truncated at the token
    limit). It is ``None`` when the provider does not report one. Truncation
    matters for grounding: a cut-off reply can drop citations or produce
    invalid structured output, so callers should treat ``"length"`` as
    unreliable.
    """

    text: str
    usage: ChatUsage
    finish_reason: str | None = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Must be deterministic for a given input.

    The interface is async because real providers do network I/O; a sync call
    would tie up a server thread per request under load.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input string.

        Contract (retrieval's cosine similarity depends on all of these):

        - **Order-preserving**: ``result[i]`` is the embedding of ``texts[i]``.
          Adapters must not rely on backend response order (sort by the
          provider's index if needed).
        - **Fixed dimension**: every vector has the same length for a given
          provider+model, so vectors are directly comparable.
        - **Deterministic**: the same input yields the same vector.
        - **Empty input**: an empty ``texts`` returns ``[]`` (no backend call).
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates a chat completion from a system + user message.

    Async for the same reason as :class:`EmbeddingProvider`: chat is network
    I/O and must not block the event loop.
    """

    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        history: Sequence[ChatMessage] | None = None,
    ) -> ChatResult:
        """Return the model's reply.

        When ``json_object`` is true the provider must constrain the model to
        emit a single valid JSON object (the grounding layer relies on this for
        structured, verifiable citations).

        ``history`` carries prior conversation turns (oldest-first) to place
        between the system message and the current ``user`` message, so the
        assistant answers with multi-turn context. It is ``None`` for a
        single-shot (stateless) call.
        """
        ...
