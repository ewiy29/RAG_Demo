"""FastAPI surface: /health, /ingest, /ask.

The pipeline is built once on startup and stored on ``app.state`` so requests
share a single vector store and provider pair. ``create_app`` accepts an
injected pipeline so tests can drive the API with the fake provider + in-memory
store, fully offline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .config import get_settings
from .documents import SUPPORTED_EXTENSIONS
from .logging_utils import configure_logging
from .pipeline import RagPipeline, build_pipeline


class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="The natural-language question.")


class CitationModel(BaseModel):
    marker: int
    source: str
    chunk_index: int
    score: float


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


class IngestResponseModel(BaseModel):
    documents: int
    chunks: int
    files: list[str]


def get_pipeline(request: Request) -> RagPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        # Build lazily on startup unless a pipeline was injected (tests).
        app.state.pipeline = pipeline or build_pipeline(settings)
        yield

    app = FastAPI(title="RAG Demo", version="0.1.0", lifespan=lifespan)
    # If injected, also set immediately so TestClient works without triggering
    # a real provider build during import.
    if pipeline is not None:
        app.state.pipeline = pipeline

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "provider": settings.provider}

    @app.post("/ingest", response_model=IngestResponseModel)
    async def ingest(
        files: list[UploadFile],
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> IngestResponseModel:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        for f in files:
            name = Path(f.filename or "").name
            ext = Path(name).suffix.lower()
            if not name or ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file {f.filename!r}. "
                    f"Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
                )
            dest = upload_dir / name
            dest.write_bytes(await f.read())
            saved.append(str(dest))

        if not saved:
            raise HTTPException(status_code=400, detail="No files provided")

        result = pipe.ingest(saved)
        return IngestResponseModel(
            documents=result.documents, chunks=result.chunks, files=saved
        )

    @app.post("/ask", response_model=AskResponseModel)
    def ask(
        body: AskRequest,
        pipe: RagPipeline = Depends(get_pipeline),
    ) -> AskResponseModel:
        resp = pipe.ask(body.query)
        return AskResponseModel(
            query=resp.query,
            answer=resp.answer,
            grounded=resp.grounded,
            citations=[
                CitationModel(
                    marker=c.marker,
                    source=c.source,
                    chunk_index=c.chunk_index,
                    score=c.score,
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

    return app


# Module-level app for `uvicorn rag.api:app`. The pipeline is built on startup
# (in the lifespan handler), not at import time, so importing this module never
# requires an API key.
app = create_app()
