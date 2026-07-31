"""Retrieval: embed the query, fetch top-k, apply a minimum-score threshold.

The threshold is the guardrail's first line of defence: if no chunk is at least
``min_score`` similar to the query, retrieval returns an empty list and the
generator refuses to answer rather than grounding on irrelevant text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .providers.base import ChatMessage, EmbeddingProvider, LLMProvider
from .vectorstore.base import VectorStore

# System instruction for the multi-turn query-condensation step. The model sees
# the prior turns as conversation history and rewrites the final follow-up into
# a self-contained question, so retrieval (which embeds the query) is not
# starved of the earlier context.
CONDENSE_SYSTEM = (
    "You rewrite a user's follow-up question into a standalone search query. "
    "Given the conversation history and the follow-up, produce a single "
    "self-contained question that preserves the subject and intent of the "
    "follow-up but needs no prior context to understand. Output ONLY the "
    "rewritten question, with no preamble, quotes, or explanation."
)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float


async def condense_query(
    query: str,
    history: Sequence[ChatMessage] | None,
    llm: LLMProvider,
) -> str:
    """Fold prior turns into a standalone query *before* embedding.

    A bare follow-up ("what about its cost?") embeds poorly because retrieval
    has no earlier subject to match on. With conversation ``history`` present we
    ask the LLM to rewrite the follow-up into a self-contained question; with no
    history there is nothing to fold, so the query is returned unchanged (no LLM
    call). The ``Follow-up:`` label is a lockstep contract with the fake
    provider's condensation stub.
    """

    if not history:
        return query
    result = await llm.chat(
        system=CONDENSE_SYSTEM,
        user=f"Follow-up: {query}",
        json_object=False,
        history=history,
    )
    rewritten = result.text.strip()
    return rewritten or query


async def retrieve(
    query: str,
    store: VectorStore,
    embedder: EmbeddingProvider,
    k: int,
    min_score: float,
    *,
    user_id: str,
    history: Sequence[ChatMessage] | None = None,
    llm: LLMProvider | None = None,
) -> list[RetrievedChunk]:
    """Return up to ``k`` chunks scoring at least ``min_score``, best first.

    Retrieval is scoped to ``user_id`` via a metadata filter, so one tenant can
    never retrieve another tenant's uploaded chunks.

    For multi-turn chat, pass ``history`` + ``llm`` and the query is condensed
    into a standalone form (folding in prior turns) *before* it is embedded.
    """

    search_query = query
    if history and llm is not None:
        search_query = await condense_query(query, history, llm)

    query_embedding = (await embedder.embed([search_query]))[0]
    hits = await store.query(query_embedding, k=k, where={"user_id": user_id})

    results: list[RetrievedChunk] = []
    for hit in hits:
        if hit.score < min_score:
            continue
        results.append(
            RetrievedChunk(
                id=hit.id,
                text=hit.text,
                source=hit.metadata.source,
                chunk_index=hit.metadata.chunk_index,
                score=hit.score,
            )
        )
    return results
