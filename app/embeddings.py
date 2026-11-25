"""Turn text into unit-length vectors, so that dot product == cosine similarity."""
from __future__ import annotations

import logging
import math
import zlib
from collections import Counter
from typing import Protocol

import numpy as np

from .textutil import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str                     # saved with the index so we notice when the model changes
    dim: int
    recommended_min_score: float  # similarity below this is treated as "not relevant"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim) with L2-normalised rows."""
        ...


class SentenceTransformerEmbedder:
    """Dense semantic embeddings from a Sentence-Transformers model (runs on PyTorch)."""

    recommended_min_score = 0.2

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import: load lazily

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.encode(["probe"]).shape[1])

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim)


class HashingEmbedder:
    """Dependency-free lexical embedder built on the hashing trick.

    Every word and word-pair is hashed into one of `dim` buckets (with a +/- sign
    to reduce collision bias) and weighted by 1 + log(count). It only matches
    shared words, not meaning, so use it for tests, CI and offline demos rather
    than for production quality retrieval.
    """

    # Lexical scores for a relevant chunk and for an unrelated one overlap a lot, so this cut-off is
    # deliberately permissive: better to let a borderline question through (the answerer still refuses
    # when no sentence shares a word with it) than to reject legitimate ones.
    recommended_min_score = 0.05

    def __init__(self, dim: int = 4096):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = tokenize(text)
            features = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]  # word pairs
            for feature, count in Counter(features).items():
                h = zlib.crc32(feature.encode("utf-8"))  # stable across runs, unlike hash()
                sign = 1.0 if (h >> 16) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(count))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


def build_embedder(backend: str, model_name: str) -> Embedder:
    """Create the embedder named by the settings ('auto' falls back to hashing)."""
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    try:  # backend == "auto"
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # not installed, or the model could not be downloaded
        log.warning("Sentence-Transformers unavailable (%s); using the hashing embedder.", exc)
        return HashingEmbedder()
