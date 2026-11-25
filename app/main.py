"""The HTTP API: a thin FastAPI layer over RagPipeline.

Run it with:  uvicorn app.main:app --reload
Interactive docs (try every endpoint in the browser) are served at /docs.

Route functions are plain `def`, not `async def`, on purpose: embedding text and searching
the index are blocking, CPU-bound calls, so FastAPI runs them in its thread pool instead of
freezing the event loop for every other request.
"""
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, Security, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader

from . import __version__
from .config import Settings
from .errors import (
    DocQAError,
    DocumentNotFound,
    DocumentParseError,
    EmptyDocument,
    FileTooLarge,
    LLMError,
    UnsupportedFileType,
)
from .rag import Hit, RagPipeline, doc_id_for
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    Source,
)

log = logging.getLogger("docqa.api")

DESCRIPTION = """
Upload PDF, DOCX, TXT or MD files, then ask questions about them. Answers are generated
**only from the retrieved passages** and come with the sources they were taken from.

1. `POST /documents` - upload a file (parsed, cleaned, chunked, embedded, indexed).
2. `POST /ask` - ask a question; get an answer plus cited sources.
3. `POST /search` - see the raw retrieved chunks (useful for debugging retrieval).

If the server was started with an `API_KEY`, click **Authorize** and enter it (header `X-API-Key`).
"""

# Domain error -> HTTP status. Anything not listed is a generic 400.
HTTP_STATUS: dict[type[DocQAError], int] = {
    UnsupportedFileType: 415,
    FileTooLarge: 413,
    DocumentParseError: 422,
    EmptyDocument: 422,
    DocumentNotFound: 404,
    LLMError: 502,
}

# A multipart upload is slightly bigger than the file inside it (boundaries, headers).
_MULTIPART_OVERHEAD = 64 * 1024


# ---- dependencies ------------------------------------------------------------------------
def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, supplied: str | None = Security(_api_key_header)) -> None:
    """When the server has an API_KEY configured, every protected route needs a matching header."""
    expected = request.app.state.settings.api_key
    if expected and not (supplied and secrets.compare_digest(supplied.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# ---- helpers -----------------------------------------------------------------------------
def _safe_filename(raw: str | None) -> str:
    """Keep only the file name: browsers send one, but other clients may send a full path."""
    name = re.split(r"[\\/]", raw or "")[-1].strip()
    return name or "upload"


def _to_source(hit: Hit) -> Source:
    record = hit.record
    return Source(
        doc_id=record.doc_id,
        filename=record.filename,
        chunk_id=record.chunk_id,
        page=record.page,
        score=round(hit.score, 4),
        text=record.text,
    )


# ---- routes ------------------------------------------------------------------------------
public = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


@public.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@public.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: PipelineDep) -> HealthResponse:
    """Liveness check (no API key needed) that also reports what the service is running with."""
    return HealthResponse(
        status="ok",
        embedder=pipeline.embedder.name,
        llm=pipeline.llm.name,
        documents=len(pipeline.documents()),
        chunks=len(pipeline.store),
    )


@protected.post(
    "/documents",
    response_model=DocumentInfo,
    status_code=201,
    tags=["documents"],
    summary="Upload a PDF, DOCX, TXT or MD file",
)
def upload_document(file: UploadFile, request: Request, response: Response, pipeline: PipelineDep) -> DocumentInfo:
    """Parse, clean, chunk, embed and index the file.

    Re-uploading identical bytes is safe: the existing document is returned with HTTP 200
    instead of creating a duplicate.
    """
    max_mb = request.app.state.settings.max_upload_mb
    limit = max_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # one extra byte tells us the file is over the limit
    if len(data) > limit:
        raise FileTooLarge(f"File is larger than the {max_mb} MB limit.")
    if not data:
        raise EmptyDocument("The uploaded file is empty.")

    already_stored = pipeline.store.get_document(doc_id_for(data)) is not None
    doc = pipeline.ingest(_safe_filename(file.filename), data)
    if already_stored:
        response.status_code = 200
    return DocumentInfo(doc_id=doc.doc_id, filename=doc.filename, num_chunks=doc.num_chunks)


@protected.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(pipeline: PipelineDep) -> list[DocumentInfo]:
    return [DocumentInfo(doc_id=d.doc_id, filename=d.filename, num_chunks=d.num_chunks) for d in pipeline.documents()]


@protected.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
def delete_document(doc_id: str, pipeline: PipelineDep) -> Response:
    pipeline.delete(doc_id)
    return Response(status_code=204)


@protected.post("/ask", response_model=AskResponse, tags=["question answering"])
def ask(body: AskRequest, pipeline: PipelineDep) -> AskResponse:
    """Retrieve the most relevant chunks, then answer using only those chunks.

    `grounded` is false when nothing relevant was found; in that case no language model is called.
    """
    answer = pipeline.ask(body.question, body.top_k, body.doc_id)
    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        answered_by=answer.answered_by,
        sources=[_to_source(hit) for hit in answer.hits],
    )


@protected.post("/search", response_model=SearchResponse, tags=["question answering"])
def search(body: SearchRequest, pipeline: PipelineDep) -> SearchResponse:
    """Return the raw top-k chunks with their similarity scores (no answer is generated)."""
    hits = pipeline.search(body.query, body.top_k, body.doc_id)
    return SearchResponse(results=[_to_source(hit) for hit in hits])


# ---- application factory -----------------------------------------------------------------
def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    """Build the application.

    In production both arguments are omitted and everything comes from environment variables
    (plus a local .env file). Tests pass their own settings and pipeline.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = settings
        if config is None:
            load_dotenv()  # picks up a local .env file; variables already in the environment win
            config = Settings.from_env()
        app.state.settings = config
        app.state.pipeline = pipeline or RagPipeline.from_settings(config)  # fail fast if misconfigured
        yield

    app = FastAPI(title="DocQA - Document Q&A", version=__version__, description=DESCRIPTION, lifespan=lifespan)
    app.include_router(public)
    app.include_router(protected)

    @app.exception_handler(DocQAError)
    async def handle_domain_error(request: Request, exc: DocQAError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=HTTP_STATUS.get(type(exc), 400))

    @app.middleware("http")
    async def guard_and_time(request: Request, call_next):
        started = time.perf_counter()
        if request.method == "POST" and request.url.path == "/documents":
            max_mb = request.app.state.settings.max_upload_mb
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > max_mb * 1024 * 1024 + _MULTIPART_OVERHEAD:
                # Refuse before the body is read, so a huge upload cannot exhaust memory or disk.
                return JSONResponse({"detail": f"File is larger than the {max_mb} MB limit."}, status_code=413)
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.1f}"
        log.info("%s %s -> %d in %.0f ms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    return app


app = create_app()
