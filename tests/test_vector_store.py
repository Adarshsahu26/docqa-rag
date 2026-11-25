"""VectorStore: search, filtering, deletion and persistence, on both the FAISS and the numpy back end."""
import numpy as np
import pytest

from app.vector_store import VectorStore
from tests.helpers import rec

DIM = 4


def unit(i: int) -> np.ndarray:
    """The i-th basis vector: unit length, and orthogonal to every other basis vector."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[i] = 1.0
    return vector


@pytest.fixture(params=["faiss", "numpy"])
def use_faiss(request) -> bool:
    if request.param == "faiss":
        pytest.importorskip("faiss")
    return request.param == "faiss"


def fill(store: VectorStore) -> VectorStore:
    store.add(
        np.stack([unit(0), unit(1), unit(2)]),
        [rec("a", 0, "zero"), rec("a", 1, "one"), rec("b", 0, "two")],
    )
    return store


@pytest.fixture()
def store(use_faiss) -> VectorStore:
    return fill(VectorStore(DIM, "test-embedder", use_faiss=use_faiss))


# ---- searching ---------------------------------------------------------------------------------
def test_the_closest_chunk_comes_first_with_its_cosine_score(store):
    hits = store.search(unit(1), k=3)
    assert hits[0][0].text == "one"
    assert hits[0][1] == pytest.approx(1.0)
    assert [round(score, 3) for _, score in hits[1:]] == [0.0, 0.0]


def test_k_limits_the_number_of_results_and_may_exceed_the_store_size(store):
    assert len(store.search(unit(0), k=2)) == 2
    assert len(store.search(unit(0), k=50)) == 3


def test_a_non_positive_k_or_an_empty_store_returns_nothing(use_faiss):
    empty = VectorStore(DIM, "test-embedder", use_faiss=use_faiss)
    assert empty.search(unit(0), k=3) == []
    assert fill(empty).search(unit(0), k=0) == []


def test_restricting_to_a_document_ignores_better_matches_elsewhere(store):
    # "zero" (document a) is the best match overall, but we only want document b.
    hits = store.search(unit(0), k=1, doc_id="b")
    assert [record.text for record, _ in hits] == ["two"]


def test_restricting_to_an_unknown_document_returns_nothing(store):
    assert store.search(unit(0), k=3, doc_id="nope") == []


def test_a_query_of_the_wrong_size_is_rejected(store):
    with pytest.raises(ValueError, match="dimension"):
        store.search(np.ones(7, dtype=np.float32), k=1)


def test_faiss_and_numpy_return_the_same_ranking():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(60, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = [rec(f"d{i % 5}", i, f"t{i}") for i in range(60)]
    with_faiss, with_numpy = VectorStore(16, "x", use_faiss=True), VectorStore(16, "x", use_faiss=False)
    for store in (with_faiss, with_numpy):
        store.add(vectors, records)
    query = vectors[7] + 0.1
    a, b = with_faiss.search(query, 10), with_numpy.search(query, 10)
    assert [r.text for r, _ in a] == [r.text for r, _ in b]
    assert [s for _, s in a] == pytest.approx([s for _, s in b], abs=1e-5)


def test_backend_name_reports_what_is_in_use(use_faiss):
    assert VectorStore(DIM, "x", use_faiss=use_faiss).backend == ("faiss" if use_faiss else "numpy")


# ---- writing -----------------------------------------------------------------------------------
def test_vectors_with_the_wrong_shape_are_rejected(use_faiss):
    store = VectorStore(DIM, "x", use_faiss=use_faiss)
    with pytest.raises(ValueError):
        store.add(np.ones((2, 7), dtype=np.float32), [rec(), rec()])  # wrong dimension
    with pytest.raises(ValueError):
        store.add(np.ones((2, DIM), dtype=np.float32), [rec()])  # 2 vectors, 1 record
    assert len(store) == 0


def test_deleting_a_document_removes_all_of_its_chunks(store):
    assert store.delete_document("a") == 2
    assert len(store) == 1
    assert store.get_document("a") is None
    assert [record.text for record, _ in store.search(unit(0), k=5)] == ["two"]


def test_deleting_an_unknown_document_removes_nothing(store):
    assert store.delete_document("nope") == 0
    assert len(store) == 3


def test_documents_are_listed_with_their_chunk_counts(store):
    assert [(d.doc_id, d.filename, d.num_chunks) for d in store.documents()] == [
        ("a", "a.txt", 2),
        ("b", "b.txt", 1),
    ]
    assert store.get_document("b").num_chunks == 1


def test_reindex_replaces_every_vector_and_the_embedder_identity(store):
    def embed(texts):
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    store.reindex("other-model", 3, embed)
    assert (store.dim, store.embedder_name) == (3, "other-model")
    assert len(store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)) == 3


# ---- persistence -------------------------------------------------------------------------------
def test_a_saved_store_comes_back_identical(tmp_path, use_faiss):
    fill(VectorStore(DIM, "emb-1", path=tmp_path, use_faiss=use_faiss)).save()

    # dim and embedder name come from the files on disk, not from the constructor arguments
    reloaded = VectorStore(99, "ignored", path=tmp_path, use_faiss=use_faiss)
    assert (reloaded.dim, reloaded.embedder_name, len(reloaded)) == (DIM, "emb-1", 3)
    assert reloaded.search(unit(1), k=1)[0][0].text == "one"
    assert reloaded.get_document("b").filename == "b.txt"


def test_saving_leaves_exactly_three_files_and_no_temporary_ones(tmp_path):
    fill(VectorStore(DIM, "emb", path=tmp_path / "nested" / "index")).save()
    names = sorted(p.name for p in (tmp_path / "nested" / "index").iterdir())
    assert names == ["meta.json", "records.json", "vectors.npy"]


def test_an_in_memory_store_never_writes_anything(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fill(VectorStore(DIM, "emb")).save()
    assert list(tmp_path.iterdir()) == []


def test_files_that_do_not_belong_together_are_detected(tmp_path):
    fill(VectorStore(DIM, "emb", path=tmp_path)).save()
    (tmp_path / "records.json").write_text("[]", encoding="utf-8")  # 0 records, 3 vectors
    with pytest.raises(RuntimeError, match="inconsistent"):
        VectorStore(DIM, "emb", path=tmp_path)


def test_deleting_then_saving_persists_the_deletion(tmp_path):
    store = fill(VectorStore(DIM, "emb", path=tmp_path))
    store.save()
    store.delete_document("a")
    store.save()
    assert len(VectorStore(DIM, "emb", path=tmp_path)) == 1


def test_a_document_filter_never_returns_more_than_k_hits(use_faiss):
    store = VectorStore(DIM, "test-embedder", use_faiss=use_faiss)
    store.add(np.stack([unit(0), unit(1), unit(2)]), [rec("a", i, str(i)) for i in range(3)])
    assert len(store.search(unit(0), k=2, doc_id="a")) == 2
