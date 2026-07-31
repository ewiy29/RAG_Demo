"""Unified typed-error taxonomy for the RAG service.

Every domain failure the app knows how to describe is a subclass of
``RagError`` carrying a machine-readable **code** (an enum) plus a structured
**context** dict. Crucially, errors do NOT carry user-facing prose: the API
translates any ``RagError`` into a structured ``{"error": {domain, code,
context}}`` envelope and the UI owns the wording. This keeps human phrasing out
of the backend and lets clients localise/brand messages off the stable code.

Three domains are defined here:

- ``DocumentError`` -- loading/extraction failures (raised by ``documents.py``;
  wired up as part of this workstream).
- ``ProviderError`` -- embedding/chat backend failures. Defined here so the API
  boundary can translate it, but the OpenAI adapter does not yet map its SDK
  exceptions onto it (that is provider-layer hardening, a later workstream).
- ``StoreError`` -- vector-store failures. Same story: defined now, the Chroma
  adapter starts raising it in the store workstream.

The ``INTERNAL`` domain is reserved for the API's catch-all handler, so an
unexpected exception still reaches the client as a structured envelope (with no
leaked traceback) rather than an unstructured 500.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorDomain(str, Enum):
    """Top-level grouping for a typed error."""

    DOCUMENT = "document"
    PROVIDER = "provider"
    STORE = "store"
    INTERNAL = "internal"


class DocumentErrorCode(str, Enum):
    """Why a document could not be loaded/extracted."""

    NOT_FOUND = "NOT_FOUND"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    DECODE_FAILED = "DECODE_FAILED"
    EMPTY_CONTENT = "EMPTY_CONTENT"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    TOO_LARGE = "TOO_LARGE"


class ProviderErrorCode(str, Enum):
    """Why an embedding/chat provider call failed."""

    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    AUTH = "AUTH"
    CONTEXT_TOO_LONG = "CONTEXT_TOO_LONG"
    UNAVAILABLE = "UNAVAILABLE"


class StoreErrorCode(str, Enum):
    """Why a vector-store operation failed."""

    UNAVAILABLE = "UNAVAILABLE"
    WRITE_FAILED = "WRITE_FAILED"
    QUERY_FAILED = "QUERY_FAILED"


# HTTP status for each code. Kept in one place so the mapping is auditable and
# the same code always yields the same status regardless of where it is raised.
_STATUS_BY_CODE: dict[Enum, int] = {
    DocumentErrorCode.NOT_FOUND: 404,
    DocumentErrorCode.UNSUPPORTED_TYPE: 415,
    DocumentErrorCode.DECODE_FAILED: 422,
    DocumentErrorCode.EMPTY_CONTENT: 422,
    DocumentErrorCode.EXTRACTION_FAILED: 422,
    DocumentErrorCode.TOO_LARGE: 413,
    ProviderErrorCode.RATE_LIMITED: 429,
    ProviderErrorCode.TIMEOUT: 504,
    ProviderErrorCode.UNAVAILABLE: 503,
    ProviderErrorCode.AUTH: 502,
    ProviderErrorCode.CONTEXT_TOO_LONG: 422,
    StoreErrorCode.UNAVAILABLE: 503,
    StoreErrorCode.WRITE_FAILED: 502,
    StoreErrorCode.QUERY_FAILED: 502,
}


class RagError(Exception):
    """Base class for every typed domain error.

    Subclasses fix ``domain`` and accept a ``code`` from their own enum. The
    optional ``message`` is for logs/debugging only and is never placed in the
    client envelope -- clients render wording from ``code`` + ``context``.
    """

    domain: ErrorDomain = ErrorDomain.INTERNAL

    def __init__(
        self,
        code: Enum,
        *,
        context: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        self.code = code
        self.context: dict[str, Any] = dict(context or {})
        # A concise internal description for logs; defaults to the code name.
        self.message = message or code.value
        super().__init__(f"{self.domain.value}:{code.value}")

    @property
    def http_status(self) -> int:
        """HTTP status this error maps to (500 if the code is unmapped)."""

        return _STATUS_BY_CODE.get(self.code, 500)

    def to_envelope(self) -> dict[str, Any]:
        """Render the structured, prose-free envelope the API returns."""

        return {
            "error": {
                "domain": self.domain.value,
                "code": self.code.value,
                "context": self.context,
            }
        }


class DocumentError(RagError):
    """A document could not be loaded or produced no usable text."""

    domain = ErrorDomain.DOCUMENT

    def __init__(
        self,
        code: DocumentErrorCode,
        *,
        context: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(code, context=context, message=message)


class ProviderError(RagError):
    """An embedding/chat provider call failed.

    Defined for the API boundary; the OpenAI adapter does not yet translate its
    SDK exceptions onto this (provider-layer hardening workstream).
    """

    domain = ErrorDomain.PROVIDER

    def __init__(
        self,
        code: ProviderErrorCode,
        *,
        context: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(code, context=context, message=message)


class StoreError(RagError):
    """A vector-store operation failed.

    Defined for the API boundary; the Chroma adapter starts raising it in the
    store workstream.
    """

    domain = ErrorDomain.STORE

    def __init__(
        self,
        code: StoreErrorCode,
        *,
        context: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        super().__init__(code, context=context, message=message)
