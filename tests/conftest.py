"""Shared test configuration.

Forces the fake provider and clears the settings cache so the entire suite runs
offline with no API key and no network access.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAG_PROVIDER", "fake")
os.environ.setdefault("OPENAI_API_KEY", "")


@pytest.fixture(autouse=True)
def _fake_provider_env(monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "fake")
    from rag.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
