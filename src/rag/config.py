"""Application settings, sourced from environment variables (and an optional .env).

Everything tunable lives here so the rest of the code depends on a single,
typed configuration object rather than reading os.environ directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Env vars are prefixed with ``RAG_`` unless noted."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection: "openai" for real calls, "fake" for offline/tests.
    provider: str = "openai"

    # OpenAI configuration. The API key uses its conventional env var name
    # (OPENAI_API_KEY), not the RAG_ prefix, so it matches the OpenAI SDK.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector store (Chroma) persistence.
    persist_dir: str = ".chroma"
    collection: str = "rag_demo"

    # Retrieval / chunking tuning.
    top_k: int = 4
    min_score: float = 0.2
    chunk_size: int = 800
    chunk_overlap: int = 150

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Cached so the environment is read once per process. Tests that mutate the
    environment can call ``get_settings.cache_clear()`` to force a reload.
    """

    return Settings()
