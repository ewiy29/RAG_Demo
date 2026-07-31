"""Text chunking with a configurable target size and overlap.

Chunking is word-aware: we pack whitespace-delimited words up to a character
budget rather than slicing mid-word, which keeps chunks readable and avoids
cutting tokens in half. Consecutive chunks share a trailing overlap so that a
fact spanning a boundary is still retrievable from at least one chunk.

Sizes are measured in characters. This keeps the module dependency-free (no
tokenizer) while staying a reasonable proxy for token count; the tradeoff is
discussed in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

from .documents import Document


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping, word-aware chunks.

    Args:
        text: Input text (may be empty).
        size: Target maximum chunk length in characters.
        overlap: Number of characters of trailing context to repeat at the
            start of the next chunk. Must be less than ``size``.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def joined_len(words_: list[str]) -> int:
        if not words_:
            return 0
        return sum(len(w) for w in words_) + (len(words_) - 1)

    for word in words:
        # A single word longer than the budget can't share a chunk sensibly;
        # flush what we have and emit it on its own (no overlap around it).
        if len(word) > size:
            if current:
                chunks.append(" ".join(current))
            chunks.append(word)
            current = []
            current_len = 0
            continue

        added = len(word) + (1 if current else 0)
        if current and current_len + added > size:
            chunks.append(" ".join(current))
            # Carry a trailing overlap (in characters) into the next chunk.
            tail: list[str] = []
            for w in reversed(current):
                if joined_len([w, *tail]) > overlap:
                    break
                tail.insert(0, w)
            current = tail
            current_len = joined_len(current)
        current.append(word)
        current_len = joined_len(current)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Chunk a ``Document`` into indexed ``Chunk`` objects."""

    return [
        Chunk(text=piece, source=document.source, chunk_index=i)
        for i, piece in enumerate(chunk_text(document.text, size, overlap))
    ]
