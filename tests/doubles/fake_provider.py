"""Deterministic, offline provider double used by the test suite.

Relocated under ``tests/`` (WS10): this is a **test aid, not a shipped
provider**. The production factory (``rag.providers.build_providers``) only ever
builds the real OpenAI provider; tests inject these fakes directly into the
pipeline instead of selecting them via a config string.

Design goals:
- No network, no API key.
- Embeddings whose cosine similarity reflects word overlap, so retrieval
  ranking and the score threshold behave meaningfully in tests.
- A chat model that only ever answers *from the context it is given*, so the
  grounded-answer and citation paths can be asserted deterministically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Sequence

from rag.providers.base import ChatMessage, ChatResult, ChatUsage

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DEFAULT_DIM = 256

# A small English stopword list. Real embedding models learn to down-weight
# these; our toy hashing embedder would otherwise let ubiquitous words like
# "the"/"is"/"of" dominate cosine similarity, so we drop them for embeddings.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "here", "his", "how", "i", "in", "into", "is", "it", "its", "me", "my",
    "no", "not", "of", "on", "once", "only", "or", "our", "over", "own",
    "she", "so", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too", "under",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "you", "your",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _hash_bucket(token: str, dim: int) -> int:
    # md5 (not for security) gives a stable bucket across processes, unlike the
    # salted builtin hash(); usedforsecurity=False documents that + silences
    # security linters.
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % dim


class FakeEmbeddingProvider:
    """Bag-of-words hashing embedder with L2-normalised output vectors."""

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _content_tokens(text):
            vec[_hash_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Pure CPU work, no I/O; async only to satisfy the provider interface.
        return [self._embed_one(t) for t in texts]


# Matches the numbered context markers produced by generate.py, e.g. "[2]".
_MARKER_RE = re.compile(r"^\s*\[(\d+)\]", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")
# Matches the follow-up line in retrieve.py's condensation prompt (see the
# lockstep coupling note below).
_FOLLOWUP_RE = re.compile(r"^Follow-up:\s*(.*)$", re.MULTILINE)


class FakeLLMProvider:
    """A stub chat model that echoes a grounded answer from the first context block.

    It never invents content: it locates the first ``[n]`` context block in the
    user prompt, quotes its opening sentence verbatim, and cites it with ``[n]``.
    The reply is a JSON object matching the structured contract that
    ``generate.py`` expects (``refused``/``answer``/``citations`` with verbatim
    ``quote`` fields), so the extractive citation-verification path can be
    asserted deterministically offline. If the prompt contains no context
    blocks it returns an explicit refusal (the real guardrail short-circuits
    before this in normal operation).

    Multi-turn (WS7): when ``json_object`` is false the call is a **query
    condensation** request (``retrieve.condense_query``), not answer generation.
    The stub deterministically folds the prior user turns from ``history`` into
    the follow-up question so the fake embedder retrieves as if the follow-up
    were standalone. History is ignored on the JSON generation path so grounded
    answers stay deterministic.

    Note: this stub is coupled to ``generate.py``'s context format (the ``[n]``
    markers, the ``(source: ...)`` header line, and the JSON output schema) and
    to ``retrieve.py``'s condensation prompt (the ``Follow-up:`` line). The two
    must be kept in lockstep; see the ``providers/fake_provider.py`` finding in
    NOTES.md.
    """

    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        history: Sequence[ChatMessage] | None = None,
    ) -> ChatResult:
        # Non-JSON call == query condensation (retrieve.py), not generation.
        if not json_object:
            return self._condense(user, history)

        blocks = self._split_context_blocks(user)
        if not blocks:
            payload: dict = {"refused": True, "answer": "", "citations": []}
        else:
            marker, body = blocks[0]
            # The quote is copied verbatim from the block body so the grounding
            # layer's substring verification succeeds.
            snippet = _SENTENCE_END_RE.split(body.strip(), maxsplit=1)[0].strip()
            payload = {
                "refused": False,
                "answer": f"{snippet} [{marker}]",
                "citations": [{"marker": marker, "quote": snippet}],
            }

        text = json.dumps(payload)
        usage = ChatUsage(
            prompt_tokens=len(_tokenize(user)),
            completion_tokens=len(_tokenize(text)),
        )
        return ChatResult(text=text, usage=usage, finish_reason="stop")

    @staticmethod
    def _condense(user: str, history: Sequence[ChatMessage] | None) -> ChatResult:
        """Rewrite a follow-up into a standalone query by folding in prior turns.

        Deterministic stand-in for an LLM condensation step: it prepends the
        content of prior *user* turns to the follow-up question, so a bare
        follow-up ("what about its density?") carries the earlier subject words
        and the fake embedder can retrieve it.
        """

        followups = _FOLLOWUP_RE.findall(user)
        follow_up = followups[-1].strip() if followups else user.strip()
        prior_user = [
            turn.content for turn in (history or []) if turn.role == "user"
        ]
        standalone = " ".join([*prior_user, follow_up]).strip() or follow_up
        usage = ChatUsage(
            prompt_tokens=len(_tokenize(user)),
            completion_tokens=len(_tokenize(standalone)),
        )
        return ChatResult(text=standalone, usage=usage, finish_reason="stop")

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
