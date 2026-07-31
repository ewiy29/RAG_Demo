"""Unit tests for citation formatting/verification and the generation guardrail.

These exercise the structured-JSON grounding contract: the model returns
``{"refused", "answer", "citations":[{"marker","quote"}]}`` and the generator
verifies each citation's verbatim quote against the cited chunk, requiring at
least one verified citation before marking an answer ``grounded``.
"""

from __future__ import annotations

import json

from rag.generate import (
    REFUSAL_TEXT,
    Answer,
    Citation,
    format_citation,
    format_context,
    generate_answer,
    verify_citations,
)
from rag.providers.base import ChatMessage, ChatResult, ChatUsage
from rag.retrieve import RetrievedChunk


def _chunk(marker_source: str, idx: int, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{marker_source}#{idx}",
        text=text,
        source=marker_source,
        chunk_index=idx,
        score=score,
    )


def _reply(refused: bool, answer: str, citations: list[dict]) -> str:
    return json.dumps({"refused": refused, "answer": answer, "citations": citations})


class _ScriptedLLM:
    """An LLM stub that returns a fixed reply string, for deterministic assertions."""

    def __init__(self, reply: str, finish_reason: str | None = "stop") -> None:
        self.reply = reply
        self.finish_reason = finish_reason
        self.calls = 0
        self.last_json_object: bool | None = None
        self.last_history: list[ChatMessage] | None = None

    async def chat(
        self,
        system: str,
        user: str,
        *,
        json_object: bool = False,
        history=None,
    ) -> ChatResult:
        self.calls += 1
        self.last_json_object = json_object
        self.last_history = list(history) if history is not None else None
        return ChatResult(
            text=self.reply,
            usage=ChatUsage(prompt_tokens=5, completion_tokens=3),
            finish_reason=self.finish_reason,
        )


def test_format_citation_renders_source_and_index():
    c = Citation(marker=1, source="guide.md", chunk_index=3, score=0.8, quote="q")
    assert format_citation(c) == "[1] guide.md#3"
    assert str(c) == "[1] guide.md#3"


def test_format_context_numbers_blocks_from_one():
    chunks = [_chunk("a.md", 0, "alpha"), _chunk("b.md", 1, "beta")]
    context = format_context(chunks)
    assert "[1] (source: a.md, chunk: 0)" in context
    assert "[2] (source: b.md, chunk: 1)" in context
    assert "alpha" in context and "beta" in context


def test_verify_citations_keeps_verified_in_order_without_duplicates():
    chunks = [_chunk("a.md", 0, "alpha text"), _chunk("b.md", 1, "beta text")]
    raw = [
        {"marker": 2, "quote": "beta"},
        {"marker": 2, "quote": "beta text"},  # duplicate marker -> ignored
        {"marker": 1, "quote": "alpha"},
    ]
    citations = verify_citations(raw, chunks)
    assert [c.marker for c in citations] == [2, 1]
    assert citations[0].source == "b.md"
    assert citations[1].source == "a.md"
    assert citations[0].quote == "beta"


def test_verify_citations_drops_out_of_range_marker():
    chunks = [_chunk("a.md", 0, "alpha")]
    citations = verify_citations(
        [{"marker": 5, "quote": "alpha"}, {"marker": 1, "quote": "alpha"}], chunks
    )
    assert [c.marker for c in citations] == [1]


def test_verify_citations_drops_quote_not_in_chunk():
    chunks = [_chunk("a.md", 0, "the sky is blue")]
    citations = verify_citations([{"marker": 1, "quote": "the grass is green"}], chunks)
    assert citations == []


def test_verify_citations_matches_whitespace_normalized_quote():
    chunks = [_chunk("a.md", 0, "Water is a\ncompound   of hydrogen and oxygen.")]
    # Quote differs only by whitespace runs/newlines -> still verifies.
    citations = verify_citations(
        [{"marker": 1, "quote": "Water is a compound of hydrogen and oxygen"}], chunks
    )
    assert [c.marker for c in citations] == [1]


async def test_generate_refuses_without_context_and_skips_llm():
    llm = _ScriptedLLM(_reply(False, "should never be returned", [{"marker": 1, "quote": "x"}]))
    answer = await generate_answer("anything?", [], llm)
    assert isinstance(answer, Answer)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []
    assert llm.calls == 0  # guardrail short-circuits before the model


async def test_generate_returns_grounded_answer_with_verified_citation():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM(
        _reply(False, "Cats are feline animals. [1]", [{"marker": 1, "quote": "Cats are feline animals."}])
    )
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is True
    assert answer.text == "Cats are feline animals. [1]"
    assert [c.marker for c in answer.citations] == [1]
    assert answer.citations[0].source == "cats.md"
    assert answer.citations[0].quote == "Cats are feline animals."
    assert answer.usage.total_tokens == 8
    assert answer.finish_reason == "stop"
    # The grounding layer must request structured JSON output.
    assert llm.last_json_object is True


async def test_generate_refuses_when_no_citation_survives_verification():
    # The model answers and cites, but the quote is not in the chunk -> the
    # citation is dropped, leaving zero verified citations -> refuse.
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM(
        _reply(False, "Cats can fly to the moon. [1]", [{"marker": 1, "quote": "Cats can fly to the moon."}])
    )
    answer = await generate_answer("can cats fly?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


async def test_generate_refuses_when_answer_has_no_citations():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM(_reply(False, "Cats are feline animals.", []))
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT


async def test_generate_respects_structured_refusal():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM(_reply(True, "", []))
    answer = await generate_answer("what is the stock price?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


async def test_generate_refuses_on_invalid_json():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM("this is not json at all")
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT


async def test_generate_refuses_on_empty_output():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM("   ")
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT


async def test_generate_treats_truncated_reply_as_refusal():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    # Even a well-formed, verifiable reply is rejected if it was truncated.
    llm = _ScriptedLLM(
        _reply(False, "Cats are feline animals. [1]", [{"marker": 1, "quote": "Cats are feline animals."}]),
        finish_reason="length",
    )
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT
    assert answer.finish_reason == "length"


async def test_generate_parses_json_wrapped_in_code_fence():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    fenced = "```json\n" + _reply(
        False, "Cats are feline animals. [1]", [{"marker": 1, "quote": "Cats are feline animals."}]
    ) + "\n```"
    llm = _ScriptedLLM(fenced)
    answer = await generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is True
    assert [c.marker for c in answer.citations] == [1]
