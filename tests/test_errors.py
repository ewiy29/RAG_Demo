"""Unit tests for the unified typed-error taxonomy (rag.errors)."""

from __future__ import annotations

from rag.errors import (
    DocumentError,
    DocumentErrorCode,
    ErrorDomain,
    ProviderError,
    ProviderErrorCode,
    RagError,
    StoreError,
    StoreErrorCode,
)


def test_envelope_shape_is_prose_free_code_and_context():
    err = DocumentError(
        DocumentErrorCode.UNSUPPORTED_TYPE,
        context={"source": "a.csv", "extension": ".csv"},
        message="internal-only detail",
    )
    envelope = err.to_envelope()
    assert envelope == {
        "error": {
            "domain": "document",
            "code": "UNSUPPORTED_TYPE",
            "context": {"source": "a.csv", "extension": ".csv"},
        }
    }
    # The internal message must not leak into the client envelope.
    assert "internal-only detail" not in str(envelope)


def test_subclasses_fix_their_domain():
    assert DocumentError(DocumentErrorCode.NOT_FOUND).domain is ErrorDomain.DOCUMENT
    assert ProviderError(ProviderErrorCode.TIMEOUT).domain is ErrorDomain.PROVIDER
    assert StoreError(StoreErrorCode.UNAVAILABLE).domain is ErrorDomain.STORE
    assert isinstance(DocumentError(DocumentErrorCode.NOT_FOUND), RagError)


def test_http_status_mapping_per_code():
    assert DocumentError(DocumentErrorCode.NOT_FOUND).http_status == 404
    assert DocumentError(DocumentErrorCode.UNSUPPORTED_TYPE).http_status == 415
    assert DocumentError(DocumentErrorCode.DECODE_FAILED).http_status == 422
    assert DocumentError(DocumentErrorCode.EMPTY_CONTENT).http_status == 422
    assert DocumentError(DocumentErrorCode.EXTRACTION_FAILED).http_status == 422
    assert ProviderError(ProviderErrorCode.RATE_LIMITED).http_status == 429
    assert ProviderError(ProviderErrorCode.TIMEOUT).http_status == 504
    assert ProviderError(ProviderErrorCode.UNAVAILABLE).http_status == 503
    assert ProviderError(ProviderErrorCode.AUTH).http_status == 502
    assert StoreError(StoreErrorCode.UNAVAILABLE).http_status == 503
    assert StoreError(StoreErrorCode.WRITE_FAILED).http_status == 502


def test_context_defaults_to_empty_dict_and_message_defaults_to_code():
    err = ProviderError(ProviderErrorCode.AUTH)
    assert err.context == {}
    assert err.message == "AUTH"
    assert err.to_envelope()["error"]["context"] == {}
