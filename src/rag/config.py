"""Application settings, sourced from environment variables (and an optional .env).

Everything tunable lives here so the rest of the code depends on a single,
typed configuration object rather than reading os.environ directly.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Valid Python logging level names. Built once from the stdlib so the set stays
# in step with whatever the running interpreter supports.
_VALID_LOG_LEVELS = frozenset(logging.getLevelNamesMapping())


class Settings(BaseSettings):
    """Typed settings. Env vars are prefixed with ``RAG_`` unless noted.

    Values are validated at load time so a bad configuration fails immediately
    with a clear error, rather than surfacing as a confusing failure deeper in
    the pipeline.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection. Only the real OpenAI provider ships; the offline fake
    # is a test double (tests/doubles) injected directly, never selected here.
    # Constrained so a typo fails at config-load time instead of in the factory.
    provider: Literal["openai"] = "openai"

    # OpenAI configuration. The API key uses its conventional env var name
    # (OPENAI_API_KEY), not the RAG_ prefix, so it matches the OpenAI SDK.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Provider request policy (single source of truth for the OpenAI adapter).
    # A hung connection must not block a worker forever, and transient 429/5xx
    # should be retried with the SDK's native backoff rather than failing the
    # request outright.
    request_timeout_seconds: float = Field(default=30.0, ge=0.1)
    max_retries: int = Field(default=2, ge=0)
    # OpenAI caps inputs-per-request and tokens-per-input, so a large ingest is
    # split into batches of this many texts per embeddings call.
    embedding_batch_size: int = Field(default=100, ge=1)
    # Sampling temperature for chat completion. 0 is the right default for
    # grounded RAG (deterministic, faithful); exposed so it can be tuned and so
    # models that reject an explicit 0 can be accommodated.
    chat_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Vector store (Chroma) persistence.
    persist_dir: str = ".chroma"
    collection: str = "rag_demo"

    # Largest single upload the API accepts, in bytes. Oversized files are
    # rejected per-file as a typed DocumentError (not an OOM). Default: 10 MiB.
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)

    # CORS: comma-separated list of browser origins allowed to call the API
    # (the React UI). "*" (default) allows any origin, which is fine for a local
    # demo; set explicit origins in production.
    cors_allow_origins: str = "*"

    # Retrieval / chunking tuning.
    top_k: int = Field(default=4, ge=1)
    # Scores are cosine similarity, whose natural range is [-1.0, 1.0].
    min_score: float = Field(default=0.2, ge=-1.0, le=1.0)
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    # Per-user data is ephemeral: how long (seconds) an uploaded corpus lives
    # before on-demand cleanup treats it as expired. Also the native per-key TTL
    # for conversation history in the Redis hot layer. Default: 24 hours.
    session_ttl_seconds: int = Field(default=86400, ge=1)

    # Conversation / multi-turn state (WS7).
    # Hot layer selector: "redis" is the shipped two-tier store (Redis hot +
    # durable). The in-memory dev/test double (tests/doubles) is injected
    # directly, never selected here.
    conversation_store: Literal["redis"] = "redis"
    # Redis connection for the hot layer. Unset (or unreachable) -> an in-process
    # ``fakeredis`` fallback so a clone-and-run demo needs no Redis server.
    redis_url: str = ""
    # Durable persistence tier behind the hot layer. Embedded SQLite in the demo;
    # production swaps to Postgres/Cosmos behind the same interface.
    conversation_durable_backend: Literal["sqlite"] = "sqlite"
    conversation_durable_path: str = "conversations.sqlite3"
    # How many recent messages to feed back as history on each chat turn.
    conversation_history_limit: int = Field(default=10, ge=1)

    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalise and validate the log level against the stdlib level names."""

        normalised = value.upper()
        if normalised not in _VALID_LOG_LEVELS:
            valid = ", ".join(sorted(_VALID_LOG_LEVELS))
            raise ValueError(
                f"Invalid log_level {value!r}. Must be one of: {valid}."
            )
        return normalised

    @model_validator(mode="after")
    def _validate_chunk_relationship(self) -> "Settings":
        """Overlap must be strictly smaller than the chunk size to make progress."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be less than chunk_size "
                f"(got chunk_overlap={self.chunk_overlap}, "
                f"chunk_size={self.chunk_size})."
            )
        return self


def get_settings() -> Settings:
    """Return a fresh Settings instance loaded from the environment/.env.

    This is the default-settings helper for entry points (API/CLI). It is no
    longer cached: tests and callers inject their own ``Settings`` rather than
    mutating the environment, so there is no cache to clear.
    """

    return Settings()
