"""A small persistent vector store: FAISS does the search, numpy + JSON keep it durable."""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

try:  # FAISS is optional: without it we fall back to a plain numpy dot product.
    import faiss
except ImportError:  # pragma: no cover - depends on the environment
    faiss = None

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    filename: str
    chunk_id: int
    page: int | None
    text: str


@dataclass(frozen=True)
class StoredDocument:
    doc_id: str
    filename: str
    num_chunks: int


class VectorStore:
    """Cosine-similarity search over chunk embeddings.

    Vectors must be L2-normalised, which makes inner product equal to cosine
    similarity. The numpy array and record list are the source of truth and are
    saved to disk; the FAISS index is derived from them and rebuilt after deletes.

    `dim` and `embedder_name` only matter for a fresh store: a store that already
    exists on disk keeps the values it was built with.
    """

    def __init__(
        self,
        dim: int,
        embedder_name: str,
        path: Path | None = None,
        use_faiss: bool | None = None,
    ):
        if use_faiss and faiss is None:
            raise RuntimeError("faiss-cpu is not installed (pip install faiss-cpu)")
        self.dim = dim
        self.embedder_name = embedder_name
        self._path = Path(path) if path else None
        self._use_faiss = (faiss is not None) if use_faiss is None else use_faiss
        self._lock = threading.RLock()
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._records: list[ChunkRecord] = []
        self._index = None
        if self._path and (self._path / "meta.json").exists():
            self._load()
        self._rebuild_index()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def backend(self) -> str:
        return "faiss" if self._use_faiss else "numpy"

    # ---- writing ---------------------------------------------------------
    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim or len(vectors) != len(records):
            raise ValueError(
                f"expected {len(records)} vectors of dimension {self.dim}, got shape {vectors.shape}"
            )
        with self._lock:
            self._vectors = np.vstack([self._vectors, vectors])
            self._records.extend(records)
            if self._index is not None:
                self._index.add(vectors)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document; returns how many chunks were removed."""
        with self._lock:
            keep = [i for i, r in enumerate(self._records) if r.doc_id != doc_id]
            removed = len(self._records) - len(keep)
            if removed:
                self._vectors = self._vectors[keep]
                self._records = [self._records[i] for i in keep]
                self._rebuild_index()
            return removed

    def reindex(
        self, embedder_name: str, dim: int, embed: Callable[[list[str]], np.ndarray]
    ) -> None:
        """Re-embed every stored chunk (used when the embedding model has changed)."""
        with self._lock:
            texts = [r.text for r in self._records]
            vectors = embed(texts) if texts else np.zeros((0, dim), dtype=np.float32)
            self.embedder_name, self.dim = embedder_name, dim
            self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            self._rebuild_index()

    # ---- reading ---------------------------------------------------------
    def search(
        self, query: np.ndarray, k: int, doc_id: str | None = None
    ) -> list[tuple[ChunkRecord, float]]:
        """Top-k chunks by cosine similarity, optionally restricted to one document.

        The document filter is applied after the search (post-filtering), which is
        fine for a flat index at this scale; big systems filter inside the index.
        """
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"query has dimension {q.shape[1]}, index expects {self.dim}")
        with self._lock:
            total = len(self._records)
            if total == 0 or k <= 0:
                return []
            scores, ids = self._raw_search(q, total if doc_id else min(k, total))
            hits: list[tuple[ChunkRecord, float]] = []
            for score, idx in zip(scores, ids, strict=True):
                if idx < 0:
                    continue
                record = self._records[int(idx)]
                if doc_id and record.doc_id != doc_id:
                    continue
                hits.append((record, float(score)))
                if len(hits) == k:
                    break
            return hits

    def documents(self) -> list[StoredDocument]:
        with self._lock:
            counts: dict[str, list] = {}
            for record in self._records:
                entry = counts.setdefault(record.doc_id, [record.filename, 0])
                entry[1] += 1
            return [StoredDocument(doc_id, name, n) for doc_id, (name, n) in counts.items()]

    def get_document(self, doc_id: str) -> StoredDocument | None:
        return next((d for d in self.documents() if d.doc_id == doc_id), None)

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        """Write the index to `path` (no-op for an in-memory store)."""
        if self._path is None:
            return
        with self._lock:
            self._path.mkdir(parents=True, exist_ok=True)
            records = [asdict(r) for r in self._records]
            meta = {"dim": self.dim, "embedder": self.embedder_name, "count": len(records)}
            _write_atomic(self._path / "vectors.npy", lambda f: np.save(f, self._vectors))
            _write_atomic(self._path / "records.json", lambda f: f.write(_dumps(records)))
            _write_atomic(self._path / "meta.json", lambda f: f.write(_dumps(meta)))  # written last

    def _load(self) -> None:
        assert self._path is not None
        meta = json.loads((self._path / "meta.json").read_text(encoding="utf-8"))
        vectors = np.load(self._path / "vectors.npy", allow_pickle=False)
        rows = json.loads((self._path / "records.json").read_text(encoding="utf-8"))
        records = [ChunkRecord(**row) for row in rows]
        if not len(records) == len(vectors) == meta["count"]:
            raise RuntimeError(f"Index files in {self._path} are inconsistent; delete that folder to start fresh.")
        self.dim, self.embedder_name = int(meta["dim"]), meta["embedder"]
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32).reshape(len(records), self.dim)
        self._records = records
        log.info("Loaded %d chunks from %s (embedder: %s)", len(records), self._path, self.embedder_name)

    # ---- internals -------------------------------------------------------
    def _rebuild_index(self) -> None:
        if not self._use_faiss:
            self._index = None
            return
        index = faiss.IndexFlatIP(self.dim)  # exact inner-product search
        if len(self._vectors):
            index.add(self._vectors)
        self._index = index

    def _raw_search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is not None:
            scores, ids = self._index.search(query, k)
            return scores[0], ids[0]
        sims = self._vectors @ query[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return sims[order], order


def _dumps(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _write_atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write to a temp file, then rename, so a crash never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        write(handle)
    os.replace(tmp, path)
