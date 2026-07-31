"""Unit tests for the recursive chunker: boundaries, overlap, structure, edges."""

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


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("hello", size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello", size=10, overlap=-1)


# --- size as a hard cap -----------------------------------------------------


def test_every_chunk_respects_size_budget():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 40


def test_size_is_a_hard_cap_even_with_large_overlap_and_long_words():
    # Regression for the old word-packer bug: an overlap tail plus the next word
    # could exceed ``size``. The recursive packer drops overlap units from the
    # front until the incoming unit fits, so ``size`` stays a hard cap.
    text = " ".join(f"{chr(ord('a') + i) * 12}" for i in range(6))
    chunks = chunk_text(text, size=30, overlap=25)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 30


# --- overlap ----------------------------------------------------------------


def test_no_overlap_produces_disjoint_word_sequences():
    text = " ".join(f"w{i}" for i in range(12))
    chunks = chunk_text(text, size=10, overlap=0)
    seen: list[str] = []
    for chunk in chunks:
        seen.extend(chunk.split())
    assert seen == text.split()


def test_overlap_repeats_a_trailing_sentence_between_chunks():
    # One paragraph, three sentences; a budget that fits two sentences but not
    # three forces a boundary, and the overlap carries a whole trailing sentence.
    text = "Alpha one is here. Beta two follows. Gamma three ends it."
    chunks = chunk_text(text, size=40, overlap=20)
    assert len(chunks) == 2
    for chunk in chunks:
        assert len(chunk) <= 40
    # The trailing sentence of chunk 0 is repeated at the start of chunk 1.
    assert "Beta two follows." in chunks[0]
    assert chunks[1].startswith("Beta two follows.")


def test_all_words_are_covered_in_order():
    text = " ".join(f"token{i}" for i in range(40))
    chunks = chunk_text(text, size=30, overlap=9)
    seen: list[str] = []
    for chunk in chunks:
        for word in chunk.split():
            if word not in seen:
                seen.append(word)
    assert seen == text.split()


# --- structure preservation -------------------------------------------------


def test_paragraph_structure_and_interior_newlines_preserved_when_fitting():
    text = "First line.\nSecond line.\n\nSecond paragraph here."
    chunks = chunk_text(text, size=200, overlap=0)
    assert chunks == [text]
    # Blank-line paragraph break and interior single newline both survive.
    assert "\n\n" in chunks[0]
    assert "First line.\nSecond line." in chunks[0]


def test_short_paragraphs_pack_together_up_to_budget():
    text = "Para one.\n\nPara two.\n\nPara three."
    packed = chunk_text(text, size=200, overlap=0)
    assert packed == [text]

    split = chunk_text(text, size=12, overlap=0)
    assert split == ["Para one.", "Para two.", "Para three."]


def test_sentences_pack_multiple_per_chunk_not_one_each():
    text = "Alpha one is here. Beta two follows. Gamma three ends it."
    chunks = chunk_text(text, size=40, overlap=0)
    assert chunks == [
        "Alpha one is here. Beta two follows.",
        "Gamma three ends it.",
    ]


# --- oversized single token -------------------------------------------------


def test_oversized_token_is_hard_split_into_size_bounded_pieces():
    long_token = "x" * 50
    text = f"small {long_token} tail"
    chunks = chunk_text(text, size=20, overlap=5)
    for chunk in chunks:
        assert len(chunk) <= 20
    # The neighbours survive as their own units...
    assert "small" in chunks
    assert "tail" in chunks
    # ...and the oversized token is hard-split but fully reconstructable.
    token_pieces = [chunk for chunk in chunks if set(chunk) == {"x"}]
    assert "".join(token_pieces) == long_token


# --- chunk_document ---------------------------------------------------------


def test_chunk_document_assigns_source_and_sequential_indexes():
    doc = Document(source="doc.md", text=" ".join(f"w{i}" for i in range(30)))
    chunks = chunk_document(doc, size=20, overlap=5)
    assert len(chunks) > 1
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.source == "doc.md" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
