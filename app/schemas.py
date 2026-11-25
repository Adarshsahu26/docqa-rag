"""Request/response models for the HTTP API (these are what /docs displays)."""
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str = Field(description="Stable id: first 12 hex characters of the file's SHA-256.")
    filename: str
    num_chunks: int


class Source(BaseModel):
    doc_id: str
    filename: str
    chunk_id: int
    page: int | None = Field(default=None, description="1-based page number (PDFs only).")
    score: float = Field(description="Cosine similarity between the question and this chunk.")
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve; server default if omitted.")
    doc_id: str | None = Field(default=None, description="Restrict the search to one document.")


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="False when nothing relevant was found; the LLM was not called.")
    answered_by: str | None = Field(description="Provider/model that wrote the answer; null if none was called.")
    sources: list[Source]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_id: str | None = None


class SearchResponse(BaseModel):
    results: list[Source]


class HealthResponse(BaseModel):
    status: str
    embedder: str
    llm: str
    documents: int
    chunks: int
