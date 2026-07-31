"""WS9 provider-layer hardening tests (offline, no network).

The real OpenAI adapters are exercised with an injected fake async client so we
can assert the hardening behaviour without a key or network:

- SDK exceptions are translated into the typed ``ProviderError`` taxonomy with
  the right code + HTTP status (so they reach the API as a structured envelope).
- Embeddings are batched and reassembled in input order, tolerating
  out-of-order ``index`` values from the backend.
- The factory is a registry that returns a ``Providers`` bundle of protocol-
  conforming instances and rejects an unknown provider name.
"""

from __future__ import annotations

import httpx
import openai
import pytest

from rag.config import Settings
from rag.errors import ProviderError, ProviderErrorCode
from rag.providers import (
    Providers,
    build_providers,
    register_provider,
)
from rag.providers.base import EmbeddingProvider, LLMProvider
from rag.providers.openai_provider import (
    OpenAIEmbeddingProvider,
    OpenAILLMProvider,
    _map_openai_error,
)


# --- fakes for the injected AsyncOpenAI client --------------------------------


class _EmbItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _EmbResponse:
    def __init__(self, data: list[_EmbItem]) -> None:
        self.data = data


class _FakeEmbeddings:
    """Records each call and returns one vector per input, tagged with index."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def create(self, *, model: str, input):
        self.batch_sizes.append(len(input))
        data = [_EmbItem(i, [float(len(text)), float(i)]) for i, text in enumerate(input)]
        return _EmbResponse(data)


class _FakeEmbeddingsOutOfOrder(_FakeEmbeddings):
    async def create(self, *, model: str, input):
        resp = await super().create(model=model, input=input)
        resp.data = list(reversed(resp.data))  # backend returns wrong order
        return resp


class _FakeRaising:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def create(self, **kwargs):
        raise self._exc


class _FakeEmbeddingClient:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _ChatResponse:
    def __init__(self, content: str, finish_reason: str, usage) -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = usage


class _FakeChatCompletions:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeChat:
    def __init__(self, completions) -> None:
        self.completions = completions


class _FakeChatClient:
    def __init__(self, chat) -> None:
        self.chat = chat


# --- SDK exception builders ---------------------------------------------------


def _response(status: int, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "http://test/v1")
    return httpx.Response(status, headers=headers or {}, request=request)


def _rate_limited() -> openai.RateLimitError:
    return openai.RateLimitError(
        "slow down", response=_response(429, {"retry-after": "7"}), body=None
    )


def _timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://test/v1"))


def _auth() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=_response(401), body=None)


def _permission() -> openai.PermissionDeniedError:
    return openai.PermissionDeniedError("nope", response=_response(403), body=None)


def _context_too_long() -> openai.BadRequestError:
    return openai.BadRequestError(
        "This model's maximum context length is 8192 tokens: context_length_exceeded",
        response=_response(400),
        body=None,
    )


def _bad_request() -> openai.BadRequestError:
    return openai.BadRequestError("bad param", response=_response(400), body=None)


def _connection() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://test/v1"))


# --- error mapping ------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc_factory", "expected_code", "expected_status"),
    [
        (_rate_limited, ProviderErrorCode.RATE_LIMITED, 429),
        (_timeout, ProviderErrorCode.TIMEOUT, 504),
        (_auth, ProviderErrorCode.AUTH, 502),
        (_permission, ProviderErrorCode.AUTH, 502),
        (_context_too_long, ProviderErrorCode.CONTEXT_TOO_LONG, 422),
        (_bad_request, ProviderErrorCode.UNAVAILABLE, 503),
        (_connection, ProviderErrorCode.UNAVAILABLE, 503),
    ],
)
def test_map_openai_error_codes(exc_factory, expected_code, expected_status):
    err = _map_openai_error(exc_factory(), operation="embed", model="m")
    assert isinstance(err, ProviderError)
    assert err.code is expected_code
    assert err.http_status == expected_status
    assert err.context["operation"] == "embed"
    assert err.context["model"] == "m"


def test_rate_limit_context_carries_retry_after():
    err = _map_openai_error(_rate_limited(), operation="embed", model="m")
    assert err.context.get("retry_after") == "7"


async def test_embed_translates_sdk_exception():
    client = _FakeEmbeddingClient(_FakeRaising(_timeout()))
    provider = OpenAIEmbeddingProvider(client, model="m")
    with pytest.raises(ProviderError) as excinfo:
        await provider.embed(["a"])
    assert excinfo.value.code is ProviderErrorCode.TIMEOUT


async def test_chat_translates_sdk_exception():
    client = _FakeChatClient(_FakeChat(_FakeRaising(_rate_limited())))
    provider = OpenAILLMProvider(client, model="m")
    with pytest.raises(ProviderError) as excinfo:
        await provider.chat("sys", "user")
    assert excinfo.value.code is ProviderErrorCode.RATE_LIMITED


# --- embedding batching + order robustness ------------------------------------


async def test_embed_empty_input_makes_no_call():
    embeddings = _FakeEmbeddings()
    provider = OpenAIEmbeddingProvider(
        _FakeEmbeddingClient(embeddings), model="m", batch_size=2
    )
    assert await provider.embed([]) == []
    assert embeddings.batch_sizes == []


async def test_embed_batches_and_preserves_order():
    embeddings = _FakeEmbeddings()
    provider = OpenAIEmbeddingProvider(
        _FakeEmbeddingClient(embeddings), model="m", batch_size=2
    )
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    vectors = await provider.embed(texts)
    # 5 inputs, batch_size 2 -> three calls of 2/2/1.
    assert embeddings.batch_sizes == [2, 2, 1]
    assert len(vectors) == len(texts)
    # First component of each fake vector is the input length -> confirms the
    # global input order is preserved across batches.
    assert [vec[0] for vec in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


async def test_embed_resorts_out_of_order_index():
    embeddings = _FakeEmbeddingsOutOfOrder()
    provider = OpenAIEmbeddingProvider(
        _FakeEmbeddingClient(embeddings), model="m", batch_size=10
    )
    texts = ["a", "bb", "ccc"]
    vectors = await provider.embed(texts)
    # Despite the backend returning reversed data, index-sort restores order.
    assert [vec[0] for vec in vectors] == [1.0, 2.0, 3.0]


# --- chat behaviour -----------------------------------------------------------


async def test_chat_uses_configured_temperature_and_reports_finish_reason():
    completions = _FakeChatCompletions(
        _ChatResponse("hello", "stop", _Usage(3, 4))
    )
    provider = OpenAILLMProvider(
        _FakeChatClient(_FakeChat(completions)), model="m", temperature=0.7
    )
    result = await provider.chat("sys", "user")
    assert result.text == "hello"
    assert result.finish_reason == "stop"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 4
    assert completions.calls[0]["temperature"] == 0.7


# --- factory registry + result type -------------------------------------------


def test_build_providers_returns_bundle_of_conforming_instances():
    # The key uses its conventional env-var alias (OPENAI_API_KEY).
    settings = Settings(provider="openai", OPENAI_API_KEY="sk-test")
    providers = build_providers(settings)
    assert isinstance(providers, Providers)
    assert isinstance(providers.embedder, EmbeddingProvider)
    assert isinstance(providers.llm, LLMProvider)


def test_build_providers_rejects_missing_key_via_shared_client():
    settings = Settings(provider="openai", OPENAI_API_KEY="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_providers(settings)


class _ProviderSelector:
    """Minimal duck-typed settings carrying just the fields the factory reads.

    Lets us pass a provider name that ``Settings``'s ``Literal["openai"]`` would
    reject at construction, exercising the registry lookup directly.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.openai_api_key = "sk-test"
        self.embedding_model = "e"
        self.chat_model = "c"
        self.request_timeout_seconds = 30.0
        self.max_retries = 2
        self.embedding_batch_size = 100
        self.chat_temperature = 0.0


def test_registry_extension_registers_a_builder():
    marker = object()

    @register_provider("dummy-vendor")
    def _build_dummy(settings):
        return marker

    assert build_providers(_ProviderSelector("dummy-vendor")) is marker


def test_build_providers_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_providers(_ProviderSelector("nope"))
