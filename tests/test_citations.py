"""Unit tests for citation formatting/parsing and the generation guardrail."""

from __future__ import annotations

from rag.generate import (
    REFUSAL_TEXT,
    Answer,
    Citation,
    format_citation,
    format_context,
    generate_answer,
    parse_citations,
)
from rag.providers.base import ChatResult, ChatUsage
from rag.retrieve import RetrievedChunk


def _chunk(marker_source: str, idx: int, text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{marker_source}#{idx}",
        text=text,
        source=marker_source,
        chunk_index=idx,
        score=score,
    )


class _ScriptedLLM:
    """An LLM stub that returns a fixed string, for deterministic assertions."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, system: str, user: str) -> ChatResult:
        self.calls += 1
        return ChatResult(text=self.reply, usage=ChatUsage(prompt_tokens=5, completion_tokens=3))


def test_format_citation_renders_source_and_index():
    c = Citation(marker=1, source="guide.md", chunk_index=3, score=0.8)
    assert format_citation(c) == "[1] guide.md#3"
    assert str(c) == "[1] guide.md#3"


def test_format_context_numbers_blocks_from_one():
    chunks = [_chunk("a.md", 0, "alpha"), _chunk("b.md", 1, "beta")]
    context = format_context(chunks)
    assert "[1] (source: a.md, chunk: 0)" in context
    assert "[2] (source: b.md, chunk: 1)" in context
    assert "alpha" in context and "beta" in context


def test_parse_citations_maps_markers_in_order_without_duplicates():
    chunks = [_chunk("a.md", 0, "alpha"), _chunk("b.md", 1, "beta")]
    citations = parse_citations("Beta then alpha [2] and again [2] plus [1].", chunks)
    assert [c.marker for c in citations] == [2, 1]
    assert citations[0].source == "b.md"
    assert citations[1].source == "a.md"


def test_parse_citations_ignores_out_of_range_markers():
    chunks = [_chunk("a.md", 0, "alpha")]
    citations = parse_citations("Nonsense [5] and valid [1].", chunks)
    assert [c.marker for c in citations] == [1]


def test_generate_refuses_without_context_and_skips_llm():
    llm = _ScriptedLLM("this should never be returned")
    answer = generate_answer("anything?", [], llm)
    assert isinstance(answer, Answer)
    assert answer.grounded is False
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []
    assert llm.calls == 0  # guardrail short-circuits before the model


def test_generate_returns_grounded_answer_with_citations():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM("Cats are feline animals. [1]")
    answer = generate_answer("what are cats?", chunks, llm)
    assert answer.grounded is True
    assert answer.text == "Cats are feline animals. [1]"
    assert [c.marker for c in answer.citations] == [1]
    assert answer.citations[0].source == "cats.md"
    assert answer.usage.total_tokens == 8


def test_generate_respects_model_refusal_text():
    chunks = [_chunk("cats.md", 0, "Cats are feline animals.")]
    llm = _ScriptedLLM(REFUSAL_TEXT)
    answer = generate_answer("what is the stock price?", chunks, llm)
    assert answer.grounded is False
    assert answer.citations == []
