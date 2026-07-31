"""Document loading for .md, .txt, and .pdf sources.

Loaders return a normalised ``Document`` (source label + plain text). Directory
inputs are walked recursively for supported extensions so a whole corpus folder
can be ingested in one call.

Failures are surfaced as typed :class:`~rag.errors.DocumentError` values (with a
machine-readable code, no user-facing prose) instead of bare stdlib exceptions.
This lets a batch load report *per-file* success/failure -- see
:func:`load_paths_with_results` -- rather than aborting the whole ingest on the
first bad file, which is what the multi-file upload UI needs.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DocumentError, DocumentErrorCode

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class FileFailure:
    """A single file that could not be loaded, with its typed error."""

    source: str
    error: DocumentError


@dataclass(frozen=True)
class DocumentLoadResult:
    """The outcome of loading a batch of paths: what loaded, what failed."""

    documents: list[Document] = field(default_factory=list)
    failures: list[FileFailure] = field(default_factory=list)


def _decode_text(source: str, data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A richer encoding fallback (utf-8-sig / secondary encodings) is a
        # documented follow-up; for now a non-utf-8 file is a typed failure
        # rather than a crash mid-ingest.
        raise DocumentError(
            DocumentErrorCode.DECODE_FAILED,
            context={"source": source, "encoding": "utf-8"},
            message=f"Could not decode {source} as utf-8: {exc}",
        ) from exc


def _extract_pdf(source: str, data: bytes) -> str:
    from pypdf import PdfReader

    try:
        # Read from an in-memory buffer so PDFs can be parsed straight from the
        # uploaded bytes without ever touching disk.
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentError:
        raise
    except Exception as exc:  # pypdf raises a variety of parse errors
        raise DocumentError(
            DocumentErrorCode.EXTRACTION_FAILED,
            context={"source": source},
            message=f"Could not extract text from {source}: {exc}",
        ) from exc
    return "\n\n".join(pages)


def _build_document(source: str, ext: str, data: bytes) -> Document:
    """Normalise raw bytes + extension into a ``Document``.

    The single core both the path- and bytes-based entry points funnel through,
    so extension dispatch, decoding/extraction, and the empty-content guard stay
    identical regardless of where the bytes came from. Raises a typed
    :class:`DocumentError` (``UNSUPPORTED_TYPE`` / ``DECODE_FAILED`` /
    ``EXTRACTION_FAILED`` / ``EMPTY_CONTENT``) on any failure.
    """

    if ext == ".pdf":
        text = _extract_pdf(source, data)
    elif ext in {".md", ".txt"}:
        text = _decode_text(source, data)
    else:
        raise DocumentError(
            DocumentErrorCode.UNSUPPORTED_TYPE,
            context={"source": source, "extension": ext},
            message=f"Unsupported file type {ext!r} for {source}.",
        )

    if not text.strip():
        # An empty/whitespace-only or image-only-PDF document would silently
        # ingest as a useless empty chunk; treat it as a typed failure.
        raise DocumentError(
            DocumentErrorCode.EMPTY_CONTENT,
            context={"source": source},
            message=f"No extractable text in {source}.",
        )
    return Document(source=source, text=text)


def load_document(path: str | Path) -> Document:
    """Load a single supported file into a ``Document``.

    Raises a :class:`DocumentError` with a specific code on any failure:
    ``NOT_FOUND`` (missing file), ``UNSUPPORTED_TYPE`` (bad extension),
    ``DECODE_FAILED`` (non-utf-8 text), ``EXTRACTION_FAILED`` (PDF parse), or
    ``EMPTY_CONTENT`` (no extractable text).
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise DocumentError(
            DocumentErrorCode.NOT_FOUND,
            context={"source": str(file_path)},
            message=f"Not a file: {file_path}",
        )
    # Read the raw bytes and hand them to the shared core so the path loader and
    # the in-memory loader share one normalisation path.
    return _build_document(str(file_path), file_path.suffix.lower(), file_path.read_bytes())


def load_bytes(filename: str, data: bytes) -> Document:
    """Load an in-memory upload (raw bytes + filename) into a ``Document``.

    The no-persistence entry point: an uploaded file is chunked and embedded
    straight from memory, never staged on disk. ``filename`` supplies both the
    ``Document.source`` label and the extension used to pick text vs PDF
    handling. Raises the same typed :class:`DocumentError` codes as
    :func:`load_document` (minus ``NOT_FOUND``, which is path-only).
    """

    return _build_document(filename, Path(filename).suffix.lower(), data)


def _iter_supported_files(paths: list[str] | list[Path]) -> list[tuple[Path, bool]]:
    """Expand inputs into ``(path, was_explicit)`` file entries.

    Directories are walked recursively for supported extensions. An unsupported
    file *inside a directory* is skipped (it was never requested), but an
    unsupported file passed *explicitly* is retained (``was_explicit=True``) so
    the caller can report it as a failure rather than dropping it silently.
    """

    entries: list[tuple[Path, bool]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    entries.append((child, False))
        else:
            # Explicit path: a file, an unsupported file, or a missing path --
            # all handled by load_document so each becomes a typed result.
            entries.append((path, True))
    return entries


def load_paths_with_results(
    paths: list[str] | list[Path],
) -> DocumentLoadResult:
    """Load files/directories, collecting per-file successes and failures.

    Never raises for a per-file problem: a missing/unsupported/undecodable/empty
    file becomes a :class:`FileFailure` while the rest continue to load. This is
    the partial-success contract the batch-upload flow depends on.
    """

    result = DocumentLoadResult()
    for file_path, _was_explicit in _iter_supported_files(paths):
        try:
            result.documents.append(load_document(file_path))
        except DocumentError as error:
            result.failures.append(FileFailure(source=str(file_path), error=error))
    return result


def load_uploads_with_results(
    uploads: list[tuple[str, bytes]],
) -> DocumentLoadResult:
    """Load in-memory uploads, collecting per-file successes and failures.

    The bytes-based mirror of :func:`load_paths_with_results`: each
    ``(filename, data)`` upload is normalised straight from memory (no disk
    staging), and a file that fails to load becomes a :class:`FileFailure`
    while the rest continue. This is the partial-success contract the
    multi-file upload flow depends on.
    """

    result = DocumentLoadResult()
    for filename, data in uploads:
        try:
            result.documents.append(load_bytes(filename, data))
        except DocumentError as error:
            result.failures.append(FileFailure(source=filename, error=error))
    return result


def load_paths(paths: list[str] | list[Path]) -> list[Document]:
    """Load one or more files/directories into ``Document`` objects.

    Convenience wrapper that returns only the documents and re-raises the first
    :class:`DocumentError`. Prefer :func:`load_paths_with_results` when partial
    success matters (e.g. multi-file uploads).
    """

    result = load_paths_with_results(paths)
    if result.failures:
        raise result.failures[0].error
    return result.documents
