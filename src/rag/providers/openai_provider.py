"""OpenAI-backed embedding and chat providers.

The SDK is imported lazily inside ``__init__`` so that importing this module
(and therefore the package) never requires the ``openai`` package to be present
or an API key to be set -- important for the offline test suite.
"""

from __future__ import annotations

from typing import Sequence

from .base import ChatResult, ChatUsage


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment/.env, "
                "or use RAG_PROVIDER=fake for offline use."
            )
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        # The SDK preserves input order in resp.data.
        return [item.embedding for item in resp.data]


class OpenAILLMProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment/.env, "
                "or use RAG_PROVIDER=fake for offline use."
            )
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, system: str, user: str) -> ChatResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        usage = ChatUsage()
        if resp.usage is not None:
            usage = ChatUsage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
            )
        return ChatResult(text=text, usage=usage)
