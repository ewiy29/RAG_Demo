"""Document loading for .md, .txt, and .pdf sources.

Loaders return a normalised ``Document`` (source label + plain text). Directory
inputs are walked recursively for supported extensions so a whole corpus folder
can be ingested in one call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: str | Path) -> Document:
    """Load a single supported file into a ``Document``."""

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    ext = p.suffix.lower()
    if ext == ".pdf":
        text = _load_pdf(p)
    elif ext in {".md", ".txt"}:
        text = _load_text(p)
    else:
        raise ValueError(
            f"Unsupported file type {ext!r} for {p}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return Document(source=str(p), text=text)


def load_paths(paths: list[str] | list[Path]) -> list[Document]:
    """Load one or more files/directories into ``Document`` objects.

    Directories are searched recursively; unsupported files inside a directory
    are skipped silently, but an unsupported file passed explicitly raises.
    """

    documents: list[Document] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    documents.append(load_document(child))
        elif p.is_file():
            documents.append(load_document(p))
        else:
            raise FileNotFoundError(f"Path does not exist: {p}")
    return documents
