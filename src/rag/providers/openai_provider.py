"""OpenAI-backed embedding and chat providers.

The SDK is imported lazily (inside ``build_openai_client``) so that importing
this module -- and therefore the package -- never requires the ``openai``
package to be present or an API key to be set. That keeps the offline test
suite and a bare ``import rag`` working without the dependency.

Hardening (WS9):
- One shared ``AsyncOpenAI`` client is built once (connection pooling) and
  injected into both providers, with a request ``timeout`` and native
  ``max_retries`` backoff for transient 429/5xx.
- Raw SDK exceptions never escape this adapter: they are translated into the
  typed ``ProviderError`` taxonomy so they reach the API as a structured
  envelope rather than an unstructured 500. The internal ``message`` (for logs)
  keeps the SDK detail; the client-facing ``context`` stays prose-free.
- Embeddings are batched (OpenAI caps inputs-per-request) and both embedding
  and chat results are ordered defensively by the API-provided ``index`` rather
  than trusting response order.
- Model names and the sampling temperature come from ``config`` (single source
  of truth), not adapter-local defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from ..errors import ProviderError, ProviderErrorCode
from .base import ChatMessage, ChatResult, ChatUsage

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    from openai import AsyncOpenAI


def build_openai_client(
    api_key: str,
    *,
    timeout: float,
    max_retries: int,
) -> "AsyncOpenAI":
    """Construct a single shared ``AsyncOpenAI`` client.

    The empty-key guard lives here so both providers fail the same way, and so
    the factory that shares one client across both still gets the check. The
    SDK is imported lazily so package import needs neither ``openai`` nor a key.
    """

    from openai import AsyncOpenAI

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your environment/.env."
        )
    return AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)


def _map_openai_error(exc: Exception, *, operation: str, model: str) -> ProviderError:
    """Translate an ``openai`` SDK exception into a typed ``ProviderError``.

    The concrete SDK types are imported lazily so this module still imports
    without ``openai`` installed; the mapping degrades to ``UNAVAILABLE`` if the
    import is unavailable for any reason.
    """

    context: dict[str, Any] = {"operation": operation, "model": model}
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        context["status_code"] = status_code

    try:
        import openai
    except Exception:  # pragma: no cover - openai is a hard dependency
        return ProviderError(
            ProviderErrorCode.UNAVAILABLE,
            context=context,
            message=f"{operation} failed: {exc}",
        )

    code = ProviderErrorCode.UNAVAILABLE
    if isinstance(exc, openai.RateLimitError):
        code = ProviderErrorCode.RATE_LIMITED
        retry_after = getattr(getattr(exc, "response", None), "headers", {})
        if retry_after:
            value = retry_after.get("retry-after")
            if value is not None:
                context["retry_after"] = value
    elif isinstance(exc, openai.APITimeoutError):
        code = ProviderErrorCode.TIMEOUT
    elif isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        code = ProviderErrorCode.AUTH
    elif isinstance(exc, openai.BadRequestError):
        # A too-long prompt is a distinct, actionable failure; other bad
        # requests are surfaced generically (still no leaked prose).
        detail = f"{getattr(exc, 'code', '')} {getattr(exc, 'param', '')} {exc}".lower()
        if "context_length_exceeded" in detail or "maximum context length" in detail:
            code = ProviderErrorCode.CONTEXT_TOO_LONG
        else:
            code = ProviderErrorCode.UNAVAILABLE

    return ProviderError(code, context=context, message=f"{operation} failed: {exc}")


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: "AsyncOpenAI",
        *,
        model: str,
        batch_size: int = 100,
    ) -> None:
        self._client = client
        self.model = model
        self._batch_size = max(1, batch_size)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(items), self._batch_size):
            batch = items[start : start + self._batch_size]
            try:
                resp = await self._client.embeddings.create(
                    model=self.model, input=batch
                )
            except Exception as exc:
                raise _map_openai_error(
                    exc, operation="embed", model=self.model
                ) from exc
            # Order defensively by the API-provided index rather than trusting
            # response order.
            ordered = sorted(resp.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors


class OpenAILLMProvider:
    def __init__(
        self,
        client: "AsyncOpenAI",
        *,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self.model = model
        self._temperature = temperature

    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        history: Sequence[ChatMessage] | None = None,
    ) -> ChatResult:
        kwargs: dict = {}
        if json_object:
            # Constrain the model to a single JSON object so the grounding layer
            # can verify structured citations. (The prompt must mention "json".)
            kwargs["response_format"] = {"type": "json_object"}
        # Prior turns are placed between the system message and the current
        # user message so the model answers with multi-turn context.
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history or []:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user})
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self._temperature,
                **kwargs,
            )
        except Exception as exc:
            raise _map_openai_error(exc, operation="chat", model=self.model) from exc
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = ChatUsage()
        if resp.usage is not None:
            usage = ChatUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        return ChatResult(text=text, usage=usage, finish_reason=choice.finish_reason)
