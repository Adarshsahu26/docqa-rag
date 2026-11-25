"""Test helpers: tiny fakes and file builders. Nothing in here touches the network."""
from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings
from app.embeddings import Embedder, HashingEmbedder
from app.llm import LLM
from app.rag import RagPipeline
from app.vector_store import ChunkRecord, VectorStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = ROOT / "sample_docs"

OIL_DOC = (
    b"Oil sampling\n\n"
    b"The transformer oil is sampled every 12 months by a certified technician. "
    b"Samples must reach the laboratory within 48 hours.\n"
)
PUMP_DOC = (
    b"Pump alarms\n\n"
    b"Code P-101 means low suction pressure. Check the inlet valve before restarting the pump.\n"
)


class FakeLLM:
    """Records every call and returns a canned reply, standing in for a real language model."""

    name = "fake-llm"

    def __init__(self, reply: str = "Fake answer [1]."):
        self.reply = reply
        self.calls: list[tuple[str, list[ChunkRecord]]] = []

    def generate(self, question: str, records: Sequence[ChunkRecord]) -> str:
        self.calls.append((question, list(records)))
        return self.reply


class BrokenLLM:
    """Fails with an exception whose message must never reach API clients."""

    name = "broken-llm"

    def generate(self, question: str, records: Sequence[ChunkRecord]) -> str:
        raise RuntimeError("secret internal detail: key sk-123456")


def make_settings(**overrides) -> Settings:
    """In-memory, offline settings with small chunks so tiny documents still split."""
    values = {
        "data_dir": None,
        "embedding_backend": "hashing",
        "llm_provider": "extractive",
        "chunk_size": 60,
        "chunk_overlap": 10,
    }
    values.update(overrides)
    return Settings(**values)


def make_pipeline(
    llm: LLM | None = None, settings: Settings | None = None, embedder: Embedder | None = None
) -> RagPipeline:
    settings = settings or make_settings()
    embedder = embedder or HashingEmbedder()
    store = VectorStore(embedder.dim, embedder.name, path=settings.data_dir)
    return RagPipeline(settings, embedder, store, llm or FakeLLM())


def make_pdf(pages: Sequence[str]) -> bytes:
    """A real PDF with one page per string (use plain ASCII: the built-in font is Latin-1)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def make_docx(paragraphs: Sequence[str]) -> bytes:
    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def rec(
    doc_id: str = "d1", chunk_id: int = 0, text: str = "text", filename: str | None = None, page: int | None = None
) -> ChunkRecord:
    return ChunkRecord(doc_id, filename or f"{doc_id}.txt", chunk_id, page, text)
