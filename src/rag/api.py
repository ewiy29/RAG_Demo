"""FastAPI surface: /health, /ingest, /ask, GET /documents (list a user's
ingested sources), DELETE /documents (purge all) / DELETE /documents/{source}
(per-file), and the multi-turn conversation endpoints (/chat, GET/DELETE
/conversations).

The pipeline is built once on startup and stored on ``app.state`` so requests
share a single vector store, provider pair, and conversation store.
``create_app`` accepts an injected pipeline and/or ``Settings`` so tests can
drive the API with the fake provider + in-memory stores, fully offline, with no
environment mutation.

Tenancy: every request carries a ``user_id`` (a tenant/correlation GUID, NOT
auth) supplied via the ``X-User-Id`` header. When the header is absent one is
minted server-side and echoed back (response body + header) so a fresh client
can adopt it and keep its uploaded corpus isolated to itself.

Conversations: ``/chat`` consumes an ``X-Conversation-Id`` header (minted +
echoed when absent) to thread multi-turn history; the stateless ``/ask`` path
remains for one-shot questions.

Observability: a correlation id is bound per request by a pure-ASGI middleware
(read from ``X-Correlation-Id`` or minted, then echoed back). It is injected
into every log record for the request chain (including the error handlers) via
a contextvar, so a single request's logs can be joined even though the id is
never threaded through function signatures.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .documents import SUPPORTED_EXTENSIONS
from .errors import DocumentErrorCode, ErrorDomain, RagError
from .logging_utils import (
    configure_logging,
    get_logger,
    reset_correlation_id,
    set_correlation_id,
)
from .pipeline import RagPipeline, build_pipeline

logger = get_logger("rag.api")

USER_ID_HEADER = "X-User-Id"
CONVERSATION_ID_HEADER = "X-Conversation-Id"
CORRELATION_ID_HEADER = "X-Correlation-Id"


class CorrelationIdMiddleware:
    """Bind a per-request correlation id for the whole call chain.

    A pure-ASGI middleware (not ``BaseHTTPMiddleware``) so the contextvar it
    sets is reliably visible to the endpoint and the exception handlers, which
    run within the same async context. Reads an inbound ``X-Correlation-Id``
    (so a caller/upstream id continues) or mints a GUID, and echoes it on the
    response so a client can correlate its request.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        inbound = Headers(scope=scope).get(CORRELATION_ID_HEADER, "").strip()
        correlation_id = inbound or uuid.uuid4().hex
        token = set_correlation_id(correlation_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers[CORRELATION_ID_HEADER] = correlation_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            reset_correlation_id(token)


def _resolve_user_id(request: Request) -> str:
    """Return the request's tenant id, minting a fresh GUID when absent.

    This is a correlation/tenancy key, not authentication: a client that does
    not present one is handed a new isolated identity.
    """

    provided = request.headers.get(USER_ID_HEADER, "").strip()
    return provided or uuid.uuid4().hex


def _resolve_conversation_id(request: Request) -> str:
    """Return the request's conversation id, minting a fresh GUID when absent.

    A client that omits ``X-Conversation-Id`` is starting a new conversation; we
    mint an id and echo it back so the client can continue the same thread on
    later turns.
    """

    provided = request.headers.get(CONVERSATION_ID_HEADER, "").strip()
    return provided or uuid.uuid4().hex


class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="The natural-language question.")


class CitationModel(BaseModel):
    marker: int
    source: str
    chunk_index: int
    score: float
    quote: str


class RetrievedModel(BaseModel):
    id: str
    source: str
    chunk_index: int
    score: float


class UsageModel(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AskResponseModel(BaseModel):
    query: str
    answer: str
    grounded: bool
    citations: list[CitationModel]
    retrieved: list[RetrievedModel]
    usage: UsageModel
    latency_ms: float
    user_id: str


class FileErrorModel(BaseModel):
    """A single file that could not be ingested (prose-free, code + context)."""

    source: str
    code: str
    context: dict[str, Any] = Field(default_factory=dict)


class IngestResponseModel(BaseModel):
    documents: int
    chunks: int
    files: list[str]
    failures: list[FileErrorModel] = Field(default_factory=list)
    user_id: str


class PurgeResponseModel(BaseModel):
    user_id: str
    status: str = "purged"


class DocumentModel(BaseModel):
    """One ingested source with its stored chunk count."""

    source: str
    chunks: int


class DocumentListModel(BaseModel):
    user_id: str
    documents: list[DocumentModel] = Field(default_factory=list)


class DeleteDocumentResponseModel(BaseModel):
    user_id: str
    source: str
    status: str = "deleted"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="The natural-language question.")


class ChatResponseModel(BaseModel):
    query: str
    answer: str
    grounded: bool
    citations: list[CitationModel]
    retrieved: list[RetrievedModel]
    usage: UsageModel
    latency_ms: float
    user_id: str
    conversation_id: str


class MessageModel(BaseModel):
    role: str
    content: str
    created_at: float


class HistoryResponseModel(BaseModel):
    user_id: str
    conversation_id: str
    messages: list[MessageModel]


class ConversationListModel(BaseModel):
    user_id: str
    conversations: list[str]


def get_pipeline(request: Request) -> RagPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline


def create_app(
    pipeline: RagPipeline | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    # Prefer explicitly injected settings, then the injected pipeline's own
    # settings, and only fall back to loading from the environment. This lets
    # tests drive the app entirely through injection (no env mutation).
    if settings is None:
        settings = pipeline.settings if pipeline is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        # Build lazily on startup unless a pipeline was injected (tests).
        app.state.pipeline = pipeline or build_pipeline(settings)
        yield

    app = FastAPI(title="RAG Demo", version="0.1.0", lifespan=lifespan)
    # CORS for the browser UI (React). Origins are configurable; "*" allows any
    # (fine for a local demo). We do not use cookies (identity rides the
    # X-User-Id header), so credentials stay off -- which also keeps a "*"
    # origin list spec-compliant. Expose the minted id headers so the browser
    # can read the user/conversation/correlation ids off the response.
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            USER_ID_HEADER,
            CONVERSATION_ID_HEADER,
            CORRELATION_ID_HEADER,
        ],
    )
    # Bind a correlation id per request so every log for the chain (endpoints +
    # error handlers) is joinable, and echo it back to the caller.
    app.add_middleware(CorrelationIdMiddleware)
    # If injected, also set immediately so TestClient works without triggering
    # a real provider build during import.
    if pipeline is not None:
        app.state.pipeline = pipeline

    @app.exception_handler(RagError)
    async def _handle_rag_error(request: Request, exc: RagError) -> JSONResponse:
        # Any typed domain error becomes a structured {domain, code, context}
        # envelope with the code's HTTP status. The UI owns the wording; we log
        # the internal message for debugging (never sent to the client).
        logger.warning(
            "rag_error",
            extra={
                "domain": exc.domain.value,
                "code": exc.code.value,
                "detail": exc.message,
            },
        )
        return JSONResponse(status_code=exc.http_status, content=exc.to_envelope())

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Catch-all so an unforeseen failure still reaches the client as a
        # structured envelope with no leaked prose or traceback. The full stack
        # is logged server-side for diagnosis.
        logger.error("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "domain": ErrorDomain.INTERNAL.value,
                    "code": "INTERNAL",
                    "context": {},
                }
            },
        )

    @app.get("/health")
    def health() -> dict:
        # A bare liveness signal. It deliberately does not disclose the provider
        # or other internal configuration to unauthenticated callers.
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponseModel)
    async def ingest(
        request: Request,
        response: Response,
        files: list[UploadFile],
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> IngestResponseModel:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id

        # No-persistence ingestion: uploaded files are processed entirely in
        # memory and never written to disk. Only the derived vectors (Chroma)
        # are durable; the raw upload bytes are disposable once chunked+embedded.

        # Partial success: validate each file up front. Unsupported/oversized
        # files are reported per-file instead of failing the whole batch, so a
        # multi-file upload ingests what it can.
        max_bytes = settings.max_upload_bytes
        accepted: list[tuple[str, bytes]] = []
        failures: list[FileErrorModel] = []
        for upload in files:
            name = Path(upload.filename or "").name
            ext = Path(name).suffix.lower()
            if not name or ext not in SUPPORTED_EXTENSIONS:
                failures.append(
                    FileErrorModel(
                        source=upload.filename or "",
                        code=DocumentErrorCode.UNSUPPORTED_TYPE.value,
                        context={
                            "filename": upload.filename,
                            "extension": ext,
                        },
                    )
                )
                continue
            # Reject oversized uploads per-file (memory guard at ingress). The
            # multipart parser exposes the size, so we can refuse before pulling
            # the whole file into memory.
            size = upload.size
            if size is not None and size > max_bytes:
                failures.append(
                    FileErrorModel(
                        source=upload.filename or "",
                        code=DocumentErrorCode.TOO_LARGE.value,
                        context={
                            "filename": upload.filename,
                            "size": size,
                            "max_bytes": max_bytes,
                        },
                    )
                )
                continue
            data = await upload.read()
            # Defensive: if the parser did not report a size up front, enforce
            # the cap on the bytes we actually read.
            if len(data) > max_bytes:
                failures.append(
                    FileErrorModel(
                        source=upload.filename or "",
                        code=DocumentErrorCode.TOO_LARGE.value,
                        context={
                            "filename": upload.filename,
                            "size": len(data),
                            "max_bytes": max_bytes,
                        },
                    )
                )
                continue
            accepted.append((name, data))

        result = await pipe.ingest_uploads(accepted, user_id=user_id)
        # Fold load-time failures (decode/empty/extraction) in with the
        # unsupported/oversized ones rejected before ingest.
        failures.extend(
            FileErrorModel(
                source=failure.source,
                code=failure.error.code.value,
                context=failure.error.context,
            )
            for failure in result.failures
        )
        return IngestResponseModel(
            documents=result.documents,
            chunks=result.chunks,
            files=[name for name, _ in accepted],
            failures=failures,
            user_id=user_id,
        )

    @app.post("/ask", response_model=AskResponseModel)
    async def ask(
        request: Request,
        response: Response,
        body: AskRequest,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> AskResponseModel:
        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        # /ask is stateless (single-shot). A conversation id, if present, is
        # only logged for correlation here; multi-turn history lives on /chat.
        conversation_id: Optional[str] = (
            request.headers.get(CONVERSATION_ID_HEADER, "").strip() or None
        )
        resp = await pipe.ask(
            body.query, user_id=user_id, conversation_id=conversation_id
        )
        return AskResponseModel(
            query=resp.query,
            answer=resp.answer,
            grounded=resp.grounded,
            user_id=resp.user_id,
            citations=[
                CitationModel(
                    marker=c.marker,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    score=c.score,
                    quote=c.quote,
                )
                for c in resp.citations
            ],
            retrieved=[
                RetrievedModel(
                    id=r.id,
                    source=r.source,
                    chunk_index=r.chunk_index,
                    score=r.score,
                )
                for r in resp.retrieved
            ],
            usage=UsageModel(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            ),
            latency_ms=resp.latency_ms,
        )

    @app.get("/documents", response_model=DocumentListModel)
    async def list_documents(
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> DocumentListModel:
        """List the requesting user's ingested sources (for the manage-files UI)."""

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        sources = await pipe.list_sources(user_id)
        return DocumentListModel(
            user_id=user_id,
            documents=[
                DocumentModel(source=s.source, chunks=s.chunks) for s in sources
            ],
        )

    @app.delete("/documents", response_model=PurgeResponseModel)
    async def purge_documents(
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> PurgeResponseModel:
        """Delete all of the requesting user's uploaded corpus ("delete my data")."""

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        await pipe.purge_user(user_id)
        return PurgeResponseModel(user_id=user_id)

    @app.delete(
        "/documents/{source:path}", response_model=DeleteDocumentResponseModel
    )
    async def delete_document(
        source: str,
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> DeleteDocumentResponseModel:
        """Delete a single ingested source for the requesting user.

        The source name is taken from the path (URL-encoded by the client). Only
        the caller's own chunks for that source are removed, so two users with a
        same-named file never clobber each other. Deleting a source that does not
        exist is a no-op and still returns ``deleted`` (idempotent).
        """

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        cleaned = source.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="No source provided")
        await pipe.delete_source(cleaned, user_id=user_id)
        return DeleteDocumentResponseModel(user_id=user_id, source=cleaned)

    @app.post("/chat", response_model=ChatResponseModel)
    async def chat(
        request: Request,
        response: Response,
        body: ChatRequest,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> ChatResponseModel:
        """Multi-turn chat: answer in the context of an ongoing conversation.

        Consumes the ``X-Conversation-Id`` header (minted + echoed when absent)
        so a client can keep a thread going; prior turns are folded into the
        query before retrieval and passed to the model for context.
        """

        user_id = _resolve_user_id(request)
        conversation_id = _resolve_conversation_id(request)
        response.headers[USER_ID_HEADER] = user_id
        response.headers[CONVERSATION_ID_HEADER] = conversation_id

        resp = await pipe.chat(
            body.query, user_id=user_id, conversation_id=conversation_id
        )
        return ChatResponseModel(
            query=resp.query,
            answer=resp.answer,
            grounded=resp.grounded,
            user_id=resp.user_id,
            conversation_id=resp.conversation_id,
            citations=[
                CitationModel(
                    marker=c.marker,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    score=c.score,
                    quote=c.quote,
                )
                for c in resp.citations
            ],
            retrieved=[
                RetrievedModel(
                    id=r.id,
                    source=r.source,
                    chunk_index=r.chunk_index,
                    score=r.score,
                )
                for r in resp.retrieved
            ],
            usage=UsageModel(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            ),
            latency_ms=resp.latency_ms,
        )

    @app.get("/conversations", response_model=ConversationListModel)
    async def list_conversations(
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> ConversationListModel:
        """List the requesting user's conversation ids."""

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        conversations = await pipe.list_conversations(user_id)
        return ConversationListModel(user_id=user_id, conversations=conversations)

    @app.get(
        "/conversations/{conversation_id}", response_model=HistoryResponseModel
    )
    async def conversation_history(
        conversation_id: str,
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> HistoryResponseModel:
        """Return the recent messages of one of the user's conversations."""

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        messages = await pipe.get_history(user_id, conversation_id)
        return HistoryResponseModel(
            user_id=user_id,
            conversation_id=conversation_id,
            messages=[
                MessageModel(
                    role=m.role, content=m.content, created_at=m.created_at
                )
                for m in messages
            ],
        )

    @app.delete("/conversations", response_model=PurgeResponseModel)
    async def purge_conversations(
        request: Request,
        response: Response,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> PurgeResponseModel:
        """Delete all of the requesting user's conversation history."""

        user_id = _resolve_user_id(request)
        response.headers[USER_ID_HEADER] = user_id
        await pipe.purge_conversations(user_id)
        return PurgeResponseModel(user_id=user_id)

    return app


# Module-level app for `uvicorn rag.api:app`. The pipeline is built on startup
# (in the lifespan handler), not at import time, so importing this module never
# requires an API key.
app = create_app()
