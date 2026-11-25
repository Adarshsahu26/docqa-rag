import os
import sys
import types

import numpy as np
import pytest

import app.embeddings as embeddings_module
from app.config import DEFAULT_EMBEDDING_MODEL
from app.embeddings import HashingEmbedder, SentenceTransformerEmbedder, build_embedder


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def test_vectors_have_the_right_shape_type_and_unit_length():
    vectors = HashingEmbedder(dim=256).embed(["pump vibration alarm", "transformer oil sampling"])
    assert vectors.shape == (2, 256)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_same_text_gives_the_same_vector_in_every_process():
    first = HashingEmbedder().embed(["pump vibration"])
    second = HashingEmbedder().embed(["pump vibration"])
    assert np.array_equal(first, second)


def test_related_text_scores_higher_than_unrelated_text():
    question, related, unrelated = HashingEmbedder().embed(
        [
            "How often is the oil sampled?",
            "The transformer oil is sampled every 12 months.",
            "Peak demand limit is 4.2 MW.",
        ]
    )
    assert cosine(question, related) > cosine(question, unrelated)


def test_plural_and_singular_forms_match():
    a, b = HashingEmbedder().embed(["bearing temperature", "bearings temperatures"])
    assert cosine(a, b) > 0.99


def test_text_made_only_of_stop_words_gives_a_zero_vector_not_nan():
    vector = HashingEmbedder().embed(["the of and"])[0]
    assert not np.isnan(vector).any()
    assert np.linalg.norm(vector) == 0


def test_embedding_nothing_returns_an_empty_matrix():
    assert HashingEmbedder(dim=32).embed([]).shape == (0, 32)


# ---- choosing an embedder --------------------------------------------------------------------
def _unavailable(model_name):
    raise ImportError("No module named 'sentence_transformers'")


def test_hashing_backend_is_built_without_touching_sentence_transformers():
    assert isinstance(build_embedder("hashing", "ignored"), HashingEmbedder)


def test_auto_falls_back_to_hashing_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(embeddings_module, "SentenceTransformerEmbedder", _unavailable)
    with caplog.at_level("WARNING"):
        embedder = build_embedder("auto", DEFAULT_EMBEDDING_MODEL)
    assert isinstance(embedder, HashingEmbedder)
    assert "hashing" in caplog.text.lower()


def test_explicit_sentence_transformers_never_hides_a_failure(monkeypatch):
    monkeypatch.setattr(embeddings_module, "SentenceTransformerEmbedder", _unavailable)
    with pytest.raises(ImportError):
        build_embedder("sentence-transformers", DEFAULT_EMBEDDING_MODEL)


def test_sentence_transformer_adapter_contract(monkeypatch):
    """Whatever model sits behind it, the adapter must ask for normalised vectors and report name and dim."""
    seen = {}

    class StubModel:
        def __init__(self, name):
            seen["name"] = name

        def encode(self, texts, **kwargs):
            seen["kwargs"] = kwargs
            ones = np.ones((len(texts), 8), dtype=np.float64)
            return ones / np.linalg.norm(ones, axis=1, keepdims=True)

    stub_module = types.ModuleType("sentence_transformers")
    stub_module.SentenceTransformer = StubModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)

    embedder = SentenceTransformerEmbedder("my/model")
    vectors = embedder.embed(["a", "b", "c"])

    assert (embedder.name, embedder.dim) == ("my/model", 8)
    assert vectors.shape == (3, 8)
    assert vectors.dtype == np.float32
    assert seen["name"] == "my/model"
    assert seen["kwargs"]["normalize_embeddings"] is True


@pytest.mark.integration
def test_real_sentence_transformers_model():
    """Runs the real library. By default it downloads the default model; DOCQA_TEST_ST_MODEL can point elsewhere."""
    pytest.importorskip("sentence_transformers")
    custom = os.getenv("DOCQA_TEST_ST_MODEL")
    embedder = SentenceTransformerEmbedder(custom or DEFAULT_EMBEDDING_MODEL)
    question, near, far = embedder.embed(
        [
            "How often is the transformer oil sampled?",
            "Oil samples are taken every twelve months.",
            "The cafeteria serves lunch at noon.",
        ]
    )
    assert embedder.dim == len(question) > 0
    assert np.allclose([np.linalg.norm(v) for v in (question, near, far)], 1.0, atol=1e-3)
    if custom is None:  # a tiny random test model cannot be expected to rank meaning correctly
        assert cosine(question, near) > cosine(question, far)
