"""Unit tests for document loading + per-file partial-success results."""

from __future__ import annotations

import pytest

from rag.documents import (
    DocumentLoadResult,
    load_bytes,
    load_document,
    load_paths,
    load_paths_with_results,
    load_uploads_with_results,
)
from rag.errors import DocumentError, DocumentErrorCode


def _make_pdf(body_text: str) -> bytes:
    """Assemble a tiny single-page PDF with extractable text, offsets computed.

    Building the xref offsets programmatically keeps the fixture valid without a
    hand-tuned byte layout, so pypdf can parse it straight from memory.
    """

    stream = f"BT /F1 24 Tf 20 100 Td ({body_text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % index
        pdf += obj
        pdf += b"\nendobj\n"

    xref_offset = len(pdf)
    total = len(objects) + 1
    pdf += b"xref\n0 %d\n" % total
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += ("%010d 00000 n \n" % offset).encode("latin-1")
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % total
    pdf += b"startxref\n%d\n%%%%EOF" % xref_offset
    return bytes(pdf)


def _write(tmp_path, name, text, *, encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


def test_load_document_reads_supported_text(tmp_path):
    path = _write(tmp_path, "good.md", "hello world")
    doc = load_document(path)
    assert doc.text == "hello world"
    assert doc.source == str(path)


def test_load_document_missing_file_raises_not_found(tmp_path):
    with pytest.raises(DocumentError) as excinfo:
        load_document(tmp_path / "nope.md")
    assert excinfo.value.code is DocumentErrorCode.NOT_FOUND


def test_load_document_unsupported_type_raises(tmp_path):
    path = _write(tmp_path, "data.csv", "a,b,c")
    with pytest.raises(DocumentError) as excinfo:
        load_document(path)
    assert excinfo.value.code is DocumentErrorCode.UNSUPPORTED_TYPE
    assert excinfo.value.context["extension"] == ".csv"


def test_load_document_empty_content_raises(tmp_path):
    path = _write(tmp_path, "empty.txt", "   \n\t  ")
    with pytest.raises(DocumentError) as excinfo:
        load_document(path)
    assert excinfo.value.code is DocumentErrorCode.EMPTY_CONTENT


def test_load_document_decode_failure_raises(tmp_path):
    # Bytes that are not valid utf-8.
    path = tmp_path / "latin.txt"
    path.write_bytes(b"\xff\xfe invalid utf8 \x80")
    with pytest.raises(DocumentError) as excinfo:
        load_document(path)
    assert excinfo.value.code is DocumentErrorCode.DECODE_FAILED


def test_load_paths_with_results_collects_partial_success(tmp_path):
    good = _write(tmp_path, "good.md", "usable content here")
    empty = _write(tmp_path, "empty.txt", "")
    unsupported = _write(tmp_path, "data.csv", "a,b,c")
    missing = tmp_path / "ghost.md"

    result = load_paths_with_results([good, empty, unsupported, missing])

    assert isinstance(result, DocumentLoadResult)
    # Only the good file loads.
    assert [d.source for d in result.documents] == [str(good)]
    # The other three each become a typed failure with the right code.
    failures = {f.source: f.error.code for f in result.failures}
    assert failures[str(empty)] is DocumentErrorCode.EMPTY_CONTENT
    assert failures[str(unsupported)] is DocumentErrorCode.UNSUPPORTED_TYPE
    assert failures[str(missing)] is DocumentErrorCode.NOT_FOUND


def test_load_paths_with_results_skips_unsupported_inside_directory(tmp_path):
    _write(tmp_path, "good.md", "content")
    _write(tmp_path, "ignore.csv", "a,b")  # unsupported, not explicit -> skipped
    result = load_paths_with_results([tmp_path])
    assert [d.source.endswith("good.md") for d in result.documents] == [True]
    assert result.failures == []


def test_load_paths_wrapper_raises_first_failure(tmp_path):
    _write(tmp_path, "good.md", "content")
    with pytest.raises(DocumentError):
        load_paths([tmp_path / "missing.md"])


# --- In-memory (bytes) loading -------------------------------------------------


def test_load_bytes_reads_supported_markdown():
    doc = load_bytes("notes.md", b"hello world")
    assert doc.text == "hello world"
    assert doc.source == "notes.md"


def test_load_bytes_reads_supported_text():
    doc = load_bytes("notes.txt", "caf\u00e9 au lait".encode("utf-8"))
    assert doc.text == "caf\u00e9 au lait"
    assert doc.source == "notes.txt"


def test_load_bytes_unsupported_type_raises():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes("data.csv", b"a,b,c")
    assert excinfo.value.code is DocumentErrorCode.UNSUPPORTED_TYPE
    assert excinfo.value.context["extension"] == ".csv"


def test_load_bytes_decode_failure_raises():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes("latin.txt", b"\xff\xfe invalid utf8 \x80")
    assert excinfo.value.code is DocumentErrorCode.DECODE_FAILED


def test_load_bytes_empty_content_raises():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes("empty.txt", b"   \n\t  ")
    assert excinfo.value.code is DocumentErrorCode.EMPTY_CONTENT


def test_load_bytes_extracts_pdf_text():
    doc = load_bytes("hello.pdf", _make_pdf("Hello PDF world"))
    assert "Hello" in doc.text
    assert doc.source == "hello.pdf"


def test_load_bytes_pdf_extraction_failed_raises():
    with pytest.raises(DocumentError) as excinfo:
        load_bytes("broken.pdf", b"%PDF-1.4 this is not really a pdf at all")
    assert excinfo.value.code is DocumentErrorCode.EXTRACTION_FAILED


def test_load_uploads_with_results_collects_partial_success():
    uploads = [
        ("good.md", b"usable content here"),
        ("doc.pdf", _make_pdf("Portable text here")),
        ("empty.txt", b""),
        ("data.csv", b"a,b,c"),
        ("latin.txt", b"\xff\xfe \x80"),
    ]

    result = load_uploads_with_results(uploads)

    assert isinstance(result, DocumentLoadResult)
    # The two loadable files succeed; order preserved.
    assert [d.source for d in result.documents] == ["good.md", "doc.pdf"]
    # The rest each become a typed failure with the right code.
    failures = {f.source: f.error.code for f in result.failures}
    assert failures["empty.txt"] is DocumentErrorCode.EMPTY_CONTENT
    assert failures["data.csv"] is DocumentErrorCode.UNSUPPORTED_TYPE
    assert failures["latin.txt"] is DocumentErrorCode.DECODE_FAILED
