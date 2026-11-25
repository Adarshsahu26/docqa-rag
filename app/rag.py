"""The RAG pipeline: ingest documents, retrieve relevant chunks, answer from them."""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass

from .chunking import chunk_text
from .cleaning import clean_text, drop_repeated_lines
from .config import Settings
from .embeddings import Embedder, build_embedder
from .errors import DocumentNotFound, EmptyDocument, LLMError
from .llm import LLM, NOT_FOUND_ANSWER, build_llm
from .parsing import parse_document
from .vector_store import ChunkRecord, StoredDocument, VectorStore

log = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = "No documents have been uploaded yet."


def doc_id_for(data: bytes) -> str:
    """Stable document id: the first 12 hex characters of the file's SHA-256."""
    return hashlib.sha256(data).hexdigest()[:12]


@dataclass(frozen=True)
class Hit:
    record: ChunkRecord
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    hits: list[Hit]
    grounded: bool           # False when nothing relevant was found
    answered_by: str | None  # LLM name, or None when no model was called


class RagPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, llm: LLM):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self._ingest_lock = threading.Lock()
        if store.embedder_name != embedder.name:  # the embedding model changed between runs
            log.warning(
                "Embedding model changed (%s -> %s); re-embedding %d stored chunks",
                store.embedder_name, embedder.name, len(store),
            )
            store.reindex(embedder.name, embedder.dim, embedder.embed)
            store.save()

    @classmethod
    def from_settings(cls, settings: Settings) -> RagPipeline:
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
        store = VectorStore(embedder.dim, embedder.name, path=settings.data_dir)
        llm = build_llm(settings)
        log.info("Pipeline ready: embedder=%s, vector store=%s, llm=%s", embedder.name, store.backend, llm.name)
        return cls(settings, embedder, store, llm)

    @property
    def min_score(self) -> float:
        if self.settings.min_score is not None:
            return self.settings.min_score
        return self.embedder.recommended_min_score

    # ---- documents -------------------------------------------------------
    def ingest(self, filename: str, data: bytes) -> StoredDocument:
        """Parse, clean, chunk, embed and store a file. Re-uploading the same bytes is a no-op."""
        doc_id = doc_id_for(data)
        with self._ingest_lock:
            existing = self.store.get_document(doc_id)
            if existing:
                return existing
            pages = parse_document(filename, data)
            texts = drop_repeated_lines([page.text for page in pages])
            records: list[ChunkRecord] = []
            for page, text in zip(pages, texts, strict=True):
                # Only PDFs have pages, and so only PDFs have page numbers to remove.
                cleaned = clean_text(text, strip_page_numbers=page.number is not None)
                for chunk in chunk_text(cleaned, self.settings.chunk_size, self.settings.chunk_overlap):
                    records.append(ChunkRecord(doc_id, filename, len(records), page.number, chunk))
            if not records:
                raise EmptyDocument("No readable text found (a scanned PDF? OCR is not supported).")
            self.store.add(self.embedder.embed([r.text for r in records]), records)
            self.store.save()
            log.info("Ingested %s as %s: %d chunks", filename, doc_id, len(records))
            return StoredDocument(doc_id, filename, len(records))

    def documents(self) -> list[StoredDocument]:
        return self.store.documents()

    def delete(self, doc_id: str) -> None:
        if not self.store.delete_document(doc_id):
            raise DocumentNotFound(f"No document with id '{doc_id}'.")
        self.store.save()

    # ---- retrieval and answering ------------------------------------------
    def search(self, query: str, top_k: int | None = None, doc_id: str | None = None) -> list[Hit]:
        """Return the most similar chunks (no relevance cut-off applied)."""
        if doc_id and not self.store.get_document(doc_id):
            raise DocumentNotFound(f"No document with id '{doc_id}'.")
        vector = self.embedder.embed([query])[0]
        found = self.store.search(vector, top_k or self.settings.top_k, doc_id)
        return [Hit(record, score) for record, score in found]

    def ask(self, question: str, top_k: int | None = None, doc_id: str | None = None) -> Answer:
        if len(self.store) == 0:
            return Answer(NO_DOCUMENTS_ANSWER, [], grounded=False, answered_by=None)

        # Guard: with nothing relevant retrieved, don't call the LLM at all. That saves
        # money and removes the main opening for a made-up answer.
        hits = [hit for hit in self.search(question, top_k, doc_id) if hit.score >= self.min_score]
        if not hits:
            return Answer(NOT_FOUND_ANSWER, [], grounded=False, answered_by=None)

        try:
            text = self.llm.generate(question, [hit.record for hit in hits])
        except LLMError:
            raise
        except Exception as exc:  # SDK errors: auth, rate limit, network, ...
            log.exception("LLM call failed")
            raise LLMError("The language model request failed; see the server logs.") from exc

        if text.strip() == NOT_FOUND_ANSWER:  # the model itself said the passages don't answer it
            return Answer(NOT_FOUND_ANSWER, [], grounded=False, answered_by=self.llm.name)
        return Answer(text, hits, grounded=True, answered_by=self.llm.name)
