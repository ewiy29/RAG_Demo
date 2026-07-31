"""Lightweight recursive text chunking (dependency-free).

Chunks are produced by a *recursive* splitter that tries structural boundaries
from largest to smallest and only descends when a unit still exceeds the size
budget:

    1. Paragraph  (blank line ``\\n\\n``)   — respects the author's structure and
       preserves interior whitespace/newlines for units that fit the budget.
    2. Sentence   (regex heuristic)         — dependency-free approximation.
    3. Word       (whitespace-delimited)    — the classic word packer.
    4. Hard split (raw character slices)    — last resort for a single token
       (e.g. a giant URL) that is itself longer than the budget.

At each level whole units are *packed* up to the budget rather than emitted one
per chunk (a lone sentence usually lacks context, and a thought can span
several). Consecutive chunks share a trailing overlap carried as whole units
(sentences/paragraphs/words) so a fact spanning a boundary survives in at least
one chunk.

Sizes are measured in **characters**, not tokens. This keeps the module
dependency-free (no tiktoken/nltk/spaCy) while staying a reasonable proxy for
token count; the tradeoff (a chunk that fits the char budget can still exceed a
downstream token limit for token-dense text) is accepted for the demo and noted
in the README.

Known limitations (documented deliberately rather than hidden):

* **Sentence detection is a heuristic.** The regex splits on ``.``/``!``/``?``
  followed by whitespace and an opening/upper-case character. It therefore:
    - may over-split after abbreviations ("Dr.", "e.g.", "U.S.A.") when the next
      word is capitalised;
    - correctly avoids splitting decimals/versions ("3.14", "v1.2.0"), most
      filenames/URLs ("index.html", "example.com/path") because the following
      character is a digit or lower-case letter;
    - treats an ellipsis ("...") followed by a capital as a boundary.
  A full NLP segmenter would do better but conflicts with the tokenizer-free
  design, so we choose the heuristic and document it.
* **A single oversized unit** with no boundary inside the budget (e.g. one very
  long token) is hard-split on character count as a last resort.
* **Overlap continuity breaks around a hard-split/oversized unit:** the pieces of
  an oversized unit do not carry sentence overlap with their neighbours.
* **Empty / whitespace-only input** yields no chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import Document

# Split on one or more blank lines (a blank line may carry stray spaces/tabs).
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")

# Sentence boundary heuristic: end punctuation, then whitespace, then something
# that looks like the start of a new sentence (upper-case letter, opening quote
# or bracket). Requiring the *next* char to look sentence-initial avoids
# splitting decimals ("3.14"), versions ("v1.2.0") and most URLs/filenames.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z])")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int


def _visible_len(units: list[str], join: str) -> int:
    """Length of ``join.join(units)`` without building the string."""

    if not units:
        return 0
    return sum(len(unit) for unit in units) + len(join) * (len(units) - 1)


def _split_paragraphs(text: str) -> list[str]:
    parts = (part.strip() for part in _PARAGRAPH_BOUNDARY.split(text))
    return [part for part in parts if part]


def _split_sentences(text: str) -> list[str]:
    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(text))
    return [part for part in parts if part]


def _split_words(text: str) -> list[str]:
    return text.split()


# Boundary levels tried largest -> smallest: (splitter, join string used to
# re-assemble units that are packed together at this level).
_LEVELS: list[tuple] = [
    (_split_paragraphs, "\n\n"),
    (_split_sentences, " "),
    (_split_words, " "),
]


def _overlap_tail(units: list[str], join: str, overlap: int) -> list[str]:
    """Trailing whole units of ``units`` whose combined length fits ``overlap``."""

    if overlap <= 0:
        return []
    tail: list[str] = []
    for unit in reversed(units):
        candidate = [unit, *tail]
        if _visible_len(candidate, join) > overlap:
            break
        tail = candidate
    return tail


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort: slice a single oversized token on character count."""

    return [text[start : start + size] for start in range(0, len(text), size)]


def _split_recursive(
    text: str, size: int, overlap: int, levels: list[tuple]
) -> list[str]:
    if not levels:
        return _hard_split(text, size)

    splitter, join = levels[0]
    finer_levels = levels[1:]
    units = splitter(text)

    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        # A unit that alone exceeds the budget can't be packed at this level;
        # flush the current run and descend to the next boundary level. Its
        # pieces stand alone (no overlap bridging around an oversized unit).
        if len(unit) > size:
            if current:
                chunks.append(join.join(current))
                current = []
            chunks.extend(_split_recursive(unit, size, overlap, finer_levels))
            continue

        if current and _visible_len([*current, unit], join) > size:
            chunks.append(join.join(current))
            current = _overlap_tail(current, join, overlap)
            # Drop overlap units from the front until the incoming unit fits,
            # so ``size`` stays a hard cap even after seeding the overlap.
            while current and _visible_len([*current, unit], join) > size:
                current.pop(0)
        current.append(unit)

    if current:
        chunks.append(join.join(current))
    return chunks


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping chunks using the recursive splitter.

    ``size`` is a **hard** character cap: no returned chunk exceeds it (a single
    token longer than ``size`` is hard-split). ``overlap`` is the number of
    characters of trailing context repeated at the start of the next chunk,
    carried as whole units where possible. Whitespace-only input yields ``[]``.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    if not text.strip():
        return []

    return _split_recursive(text, size, overlap, _LEVELS)


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Chunk a ``Document`` into indexed ``Chunk`` objects."""

    return [
        Chunk(text=piece, source=document.source, chunk_index=index)
        for index, piece in enumerate(chunk_text(document.text, size, overlap))
    ]
