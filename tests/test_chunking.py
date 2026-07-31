"""Unit tests for chunking: boundaries, overlap, and edge cases."""

from __future__ import annotations

import pytest

from rag.chunking import Chunk, chunk_document, chunk_text
from rag.documents import Document


def test_empty_and_whitespace_text_yields_no_chunks():
    assert chunk_text("", size=100, overlap=10) == []
    assert chunk_text("   \n\t  ", size=100, overlap=10) == []


def test_short_text_fits_in_single_chunk():
    text = "the quick brown fox"
    assert chunk_text(text, size=100, overlap=10) == ["the quick brown fox"]


def test_each_chunk_respects_size_budget():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 40


def test_overlap_repeats_trailing_words_between_consecutive_chunks():
    text = " ".join(f"w{i}" for i in range(30))
    chunks = chunk_text(text, size=20, overlap=8)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_words = prev.split()
        nxt_words = nxt.split()
        # The next chunk must begin by repeating one or more trailing words
        # from the previous chunk (the overlap).
        assert nxt_words[0] in prev_words


def test_no_overlap_produces_disjoint_word_sequences():
    text = " ".join(f"w{i}" for i in range(12))
    chunks = chunk_text(text, size=10, overlap=0)
    seen: list[str] = []
    for c in chunks:
        seen.extend(c.split())
    # With zero overlap every word appears exactly once and order is preserved.
    assert seen == text.split()


def test_all_words_are_covered_in_order():
    text = " ".join(f"token{i}" for i in range(40))
    chunks = chunk_text(text, size=30, overlap=9)
    # Reconstruct the ordered set of first-appearances; must equal the input.
    seen: list[str] = []
    for c in chunks:
        for w in c.split():
            if not seen or seen[-1] != w:
                if w not in seen:
                    seen.append(w)
    assert seen == text.split()


def test_word_longer_than_size_becomes_its_own_chunk():
    long_word = "x" * 50
    text = f"small {long_word} tail"
    chunks = chunk_text(text, size=20, overlap=5)
    assert long_word in chunks


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=-1)


def test_chunk_document_assigns_source_and_sequential_indexes():
    doc = Document(source="doc.md", text=" ".join(f"w{i}" for i in range(30)))
    chunks = chunk_document(doc, size=20, overlap=5)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source == "doc.md" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
