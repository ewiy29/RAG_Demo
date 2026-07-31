"""Deterministic, offline provider used for tests and local demos.

Design goals:
- No network, no API key.
- Embeddings whose cosine similarity reflects word overlap, so retrieval
  ranking and the score threshold behave meaningfully in tests.
- A chat model that only ever answers *from the context it is given*, so the
  grounded-answer and citation paths can be asserted deterministically.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

from .base import ChatResult, ChatUsage

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_DIM = 256


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_bucket(token: str, dim: int) -> int:
    # md5 (not for security) gives a stable bucket across processes, unlike the
    # salted builtin hash().
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim


class FakeEmbeddingProvider:
    """Bag-of-words hashing embedder with L2-normalised output vectors."""

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            vec[_hash_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


# Matches the numbered context markers produced by generate.py, e.g. "[2]".
_MARKER_RE = re.compile(r"^\s*\[(\d+)\]", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")


class FakeLLMProvider:
    """A stub chat model that echoes a grounded answer from the first context block.

    It never invents content: it locates the first ``[n]`` context block in the
    user prompt, quotes its opening sentence, and cites it with ``[n]``. If the
    prompt contains no context blocks it returns an explicit refusal (the real
    guardrail short-circuits before this in normal operation).
    """

    def chat(self, system: str, user: str) -> ChatResult:
        blocks = self._split_context_blocks(user)
        if not blocks:
            text = "I don't have that in the provided documents."
        else:
            marker, body = blocks[0]
            snippet = _SENTENCE_END_RE.split(body.strip(), maxsplit=1)[0].strip()
            text = f"{snippet} [{marker}]"

        usage = ChatUsage(
            prompt_tokens=len(_tokenize(user)),
            completion_tokens=len(_tokenize(text)),
        )
        return ChatResult(text=text, usage=usage)

    @staticmethod
    def _split_context_blocks(user: str) -> list[tuple[int, str]]:
        """Return (marker_number, block_text) for each ``[n] ...`` block."""

        matches = list(_MARKER_RE.finditer(user))
        blocks: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(user)
            body = user[start:end].strip()
            # Drop the "(source: ..., chunk: ...)" header line if present.
            lines = body.splitlines()
            if lines and lines[0].lstrip().startswith("(source:"):
                body = "\n".join(lines[1:]).strip()
            blocks.append((int(m.group(1)), body))
        return blocks
