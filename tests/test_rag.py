"""RagPipeline: ingest, retrieve and answer, using a fake language model."""
import threading
import time

import pytest

from app.embeddings import HashingEmbedder
from app.errors import DocumentNotFound, EmptyDocument, LLMError, UnsupportedFileType
from app.llm import NOT_FOUND_ANSWER
from app.rag import NO_DOCUMENTS_ANSWER, RagPipeline, doc_id_for
from tests.helpers import (
    OIL_DOC,
    PUMP_DOC,
    BrokenLLM,
    FakeLLM,
    make_docx,
    make_pdf,
    make_pipeline,
    make_settings,
)

OIL_QUESTION = "How often is the transformer oil sampled?"
OFF_TOPIC = "What is the capital of Australia?"


# ---- ingesting ---------------------------------------------------------------------------------
def test_ingest_returns_a_document_with_a_stable_content_based_id(pipeline):
    doc = pipeline.ingest("oil.txt", OIL_DOC)
    assert doc.doc_id == doc_id_for(OIL_DOC)
    assert len(doc.doc_id) == 12
    assert doc.filename == "oil.txt"
    assert doc.num_chunks == len(pipeline.store) >= 1
    assert pipeline.documents() == [doc]


def test_ingesting_identical_bytes_again_is_a_no_op_even_under_another_name(pipeline):
    first = pipeline.ingest("oil.txt", OIL_DOC)
    second = pipeline.ingest("renamed.txt", OIL_DOC)
    assert second == first
    assert len(pipeline.documents()) == 1
    assert len(pipeline.store) == first.num_chunks


def test_unsupported_files_are_rejected_before_anything_is_stored(pipeline):
    with pytest.raises(UnsupportedFileType):
        pipeline.ingest("tool.exe", b"MZ...")
    assert len(pipeline.store) == 0


@pytest.mark.parametrize(
    ("name", "data"),
    [("blank.txt", b"  \n\t \n"), ("blank.pdf", make_pdf(["", ""])), ("blank.docx", make_docx(["", "  "]))],
)
def test_files_without_any_readable_text_are_rejected(pipeline, name, data):
    with pytest.raises(EmptyDocument):
        pipeline.ingest(name, data)
    assert pipeline.documents() == []


def test_pdf_page_numbers_are_kept_so_answers_can_cite_them(pipeline):
    pdf = make_pdf(["Pump alarm P-101 means low suction pressure.", "The transformer oil is sampled every 12 months."])
    pipeline.ingest("manual.pdf", pdf)
    top = pipeline.search("transformer oil sampled", top_k=1)[0]
    assert top.record.page == 2


def test_docx_files_are_searchable(pipeline):
    pipeline.ingest("manual.docx", make_docx(["Alarm P-101 means low suction pressure.", "Check the inlet valve."]))
    assert "low suction pressure" in pipeline.search("what is alarm P-101", top_k=1)[0].record.text


def test_running_headers_and_footers_do_not_end_up_in_the_index(pipeline):
    topics = ["lubrication", "alignment", "insulation", "calibration", "ventilation"]
    pages = [
        f"ACME Maintenance Manual v2\nThe {t} section covers {t} checks.\nPage {i + 1} of 5"
        for i, t in enumerate(topics)
    ]
    pipeline.ingest("manual.pdf", make_pdf(pages))
    stored = " ".join(hit.record.text for hit in pipeline.search("ACME manual page", top_k=50))
    assert "ACME" not in stored
    assert "Page " not in stored
    assert "lubrication section covers lubrication checks" in stored


def test_chunk_size_comes_from_the_settings():
    text = (" ".join(f"word{i}" for i in range(200)) + ".").encode()
    small = make_pipeline(settings=make_settings(chunk_size=20, chunk_overlap=0))
    large = make_pipeline(settings=make_settings(chunk_size=180, chunk_overlap=0))
    assert small.ingest("long.txt", text).num_chunks == 10
    assert large.ingest("long.txt", text).num_chunks == 2


def test_the_same_file_uploaded_from_many_threads_is_indexed_once(pipeline):
    results = []

    def upload():
        results.append(pipeline.ingest("oil.txt", OIL_DOC))

    threads = [threading.Thread(target=upload) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert len(set(results)) == 1
    assert len(pipeline.documents()) == 1
    assert len(pipeline.store) == results[0].num_chunks


class _SlowEmbedder(HashingEmbedder):
    """Embedding takes long enough for a second upload to arrive while the first one is still being embedded."""

    def embed(self, texts):
        time.sleep(0.2)
        return super().embed(texts)


def test_two_uploads_of_the_same_file_at_the_same_moment_are_indexed_once():
    pipeline = make_pipeline(embedder=_SlowEmbedder())
    both_ready = threading.Barrier(2)
    results = []

    def upload():
        both_ready.wait()
        results.append(pipeline.ingest("oil.txt", OIL_DOC))

    threads = [threading.Thread(target=upload) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results[0] == results[1]
    assert len(pipeline.documents()) == 1
    assert len(pipeline.store) == results[0].num_chunks  # a second copy of the chunks would double this


# ---- retrieving --------------------------------------------------------------------------------
def test_search_ranks_by_similarity_and_applies_no_cut_off(pipeline):
    pipeline.ingest("oil.txt", OIL_DOC)
    pipeline.ingest("pump.txt", PUMP_DOC)
    hits = pipeline.search("oil sampling", top_k=50)
    assert len(hits) == len(pipeline.store)  # even zero-similarity chunks are returned
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].record.filename == "oil.txt"


def test_search_can_be_limited_to_one_document(pipeline):
    pipeline.ingest("oil.txt", OIL_DOC)
    pump = pipeline.ingest("pump.txt", PUMP_DOC)
    hits = pipeline.search("oil sampling", top_k=50, doc_id=pump.doc_id)
    assert {hit.record.filename for hit in hits} == {"pump.txt"}


def test_an_unknown_document_id_is_an_error(pipeline):
    pipeline.ingest("oil.txt", OIL_DOC)
    with pytest.raises(DocumentNotFound):
        pipeline.search(OIL_QUESTION, doc_id="missing")
    with pytest.raises(DocumentNotFound):
        pipeline.ask(OIL_QUESTION, doc_id="missing")


# ---- answering ---------------------------------------------------------------------------------
def test_asking_with_nothing_indexed_says_so_without_calling_the_model(pipeline, fake_llm):
    answer = pipeline.ask(OIL_QUESTION)
    assert answer.text == NO_DOCUMENTS_ANSWER
    assert not answer.grounded
    assert answer.hits == []
    assert fake_llm.calls == []


def test_a_relevant_question_reaches_the_model_with_only_the_relevant_chunks(pipeline, fake_llm):
    pipeline.ingest("oil.txt", OIL_DOC)
    pipeline.ingest("pump.txt", PUMP_DOC)

    answer = pipeline.ask(OIL_QUESTION)

    assert answer.grounded
    assert answer.text == fake_llm.reply
    assert answer.answered_by == "fake-llm"
    assert {hit.record.filename for hit in answer.hits} == {"oil.txt"}
    (question, records), = fake_llm.calls
    assert question == OIL_QUESTION
    assert records == [hit.record for hit in answer.hits]


def test_an_unrelated_question_is_refused_without_spending_a_model_call(pipeline, fake_llm):
    pipeline.ingest("oil.txt", OIL_DOC)
    answer = pipeline.ask(OFF_TOPIC)
    assert answer.text == NOT_FOUND_ANSWER
    assert not answer.grounded
    assert answer.hits == []
    assert answer.answered_by is None
    assert fake_llm.calls == []


def test_when_the_model_itself_says_the_passages_do_not_answer_it_no_sources_are_shown():
    pipeline = make_pipeline(FakeLLM(reply=NOT_FOUND_ANSWER))
    pipeline.ingest("oil.txt", OIL_DOC)
    answer = pipeline.ask(OIL_QUESTION)
    assert not answer.grounded
    assert answer.hits == []
    assert answer.answered_by == "fake-llm"


def test_unexpected_model_failures_are_replaced_by_a_message_that_leaks_nothing():
    pipeline = make_pipeline(BrokenLLM())
    pipeline.ingest("oil.txt", OIL_DOC)
    with pytest.raises(LLMError) as caught:
        pipeline.ask(OIL_QUESTION)
    assert "sk-123456" not in str(caught.value)
    assert "see the server logs" in str(caught.value)


def test_expected_llm_errors_pass_through_unchanged():
    class QuotaLLM:
        name = "quota-llm"

        def generate(self, question, records):
            raise LLMError("quota exceeded")

    pipeline = make_pipeline(QuotaLLM())
    pipeline.ingest("oil.txt", OIL_DOC)
    with pytest.raises(LLMError, match="quota exceeded"):
        pipeline.ask(OIL_QUESTION)


def test_a_document_filter_restricts_what_the_model_may_see(pipeline, fake_llm):
    pipeline.ingest("oil.txt", OIL_DOC)
    pump = pipeline.ingest("pump.txt", PUMP_DOC)
    answer = pipeline.ask(OIL_QUESTION, doc_id=pump.doc_id)
    assert not answer.grounded  # the pump manual says nothing about oil sampling
    assert fake_llm.calls == []


def test_top_k_limits_how_many_chunks_the_model_sees(fake_llm):
    text = " ".join(f"The oil sampling step number {i} is documented here." for i in range(30)).encode()
    pipeline = make_pipeline(fake_llm)
    pipeline.ingest("long.txt", text)
    pipeline.ask("oil sampling step", top_k=2)
    assert len(fake_llm.calls[0][1]) == 2


def test_the_relevance_cut_off_comes_from_the_settings(fake_llm):
    strict = make_pipeline(fake_llm, make_settings(min_score=0.999))
    strict.ingest("oil.txt", OIL_DOC)
    assert not strict.ask(OIL_QUESTION).grounded

    permissive = make_pipeline(fake_llm, make_settings(min_score=-1.0))
    permissive.ingest("oil.txt", OIL_DOC)
    assert permissive.ask(OFF_TOPIC).grounded  # nothing is filtered, so even this reaches the model


def test_a_chunk_that_scores_exactly_the_cut_off_is_still_used(fake_llm):
    probe = make_pipeline(fake_llm, make_settings(min_score=-1.0))
    probe.ingest("oil.txt", OIL_DOC)
    best = probe.search(OIL_QUESTION, top_k=1)[0].score  # the same text and embedder always give this same score

    exact = make_pipeline(fake_llm, make_settings(min_score=best))
    exact.ingest("oil.txt", OIL_DOC)
    assert exact.ask(OIL_QUESTION).grounded


def test_without_a_setting_the_embedders_recommended_cut_off_applies(pipeline):
    assert pipeline.min_score == HashingEmbedder.recommended_min_score


# ---- deleting ----------------------------------------------------------------------------------
def test_a_deleted_document_can_no_longer_be_found(pipeline):
    oil = pipeline.ingest("oil.txt", OIL_DOC)
    pipeline.ingest("pump.txt", PUMP_DOC)
    pipeline.delete(oil.doc_id)
    assert [doc.filename for doc in pipeline.documents()] == ["pump.txt"]
    assert not pipeline.ask(OIL_QUESTION).grounded


def test_a_deletion_survives_a_restart(tmp_path):
    settings = make_settings(data_dir=tmp_path / "index")
    first = make_pipeline(settings=settings)
    oil = first.ingest("oil.txt", OIL_DOC)
    first.ingest("pump.txt", PUMP_DOC)
    first.delete(oil.doc_id)

    second = make_pipeline(settings=settings)  # a fresh process reads only what was saved to disk
    assert [doc.filename for doc in second.documents()] == ["pump.txt"]
    assert not second.ask(OIL_QUESTION).grounded


def test_deleting_an_unknown_document_is_an_error(pipeline):
    with pytest.raises(DocumentNotFound):
        pipeline.delete("missing")


# ---- persistence and configuration ---------------------------------------------------------------
def test_the_index_survives_a_restart(tmp_path):
    settings = make_settings(data_dir=tmp_path / "index")
    first = make_pipeline(settings=settings)
    doc = first.ingest("oil.txt", OIL_DOC)

    second = make_pipeline(settings=settings)  # a fresh process would do exactly this
    assert second.documents() == [doc]
    assert second.ask(OIL_QUESTION).grounded


def test_changing_the_embedding_model_re_embeds_the_stored_text(tmp_path):
    settings = make_settings(data_dir=tmp_path / "index")
    make_pipeline(settings=settings, embedder=HashingEmbedder(dim=256)).ingest("oil.txt", OIL_DOC)

    upgraded = make_pipeline(settings=settings, embedder=HashingEmbedder(dim=512))
    assert (upgraded.store.embedder_name, upgraded.store.dim) == ("hashing-512", 512)
    assert upgraded.search(OIL_QUESTION, top_k=1)[0].record.filename == "oil.txt"

    # the re-embedded index was saved, so the next start-up has nothing left to convert
    again = make_pipeline(settings=settings, embedder=HashingEmbedder(dim=512))
    assert again.store.embedder_name == "hashing-512"
    assert len(again.store) == len(upgraded.store)


def test_from_settings_builds_the_configured_components():
    pipeline = RagPipeline.from_settings(make_settings(embedding_backend="hashing", llm_provider="extractive"))
    assert pipeline.embedder.name.startswith("hashing")
    assert pipeline.llm.name == "extractive"
    assert pipeline.store.embedder_name == pipeline.embedder.name


# ---- cleaning at ingest -------------------------------------------------------------------------
def _stored_chunks(pipeline, query):
    return {hit.record.page: hit.record.text for hit in pipeline.search(query, top_k=10)}


def test_text_is_cleaned_before_it_is_indexed(pipeline):
    pipeline.ingest("notes.txt", b"Routine mainte-\nnance of the pump is done every twelve months.\n")
    stored = _stored_chunks(pipeline, "pump maintenance")[None]
    assert "Routine maintenance of the pump" in stored
    assert "mainte-" not in stored


def test_numbers_on_a_line_of_their_own_are_kept_in_files_without_pages(pipeline):
    pipeline.ingest("figures.txt", b"2026\nValve torque\nV-12\n45\nV-13\n60\n2027\n")
    assert _stored_chunks(pipeline, "valve torque V-13")[None] == "2026\nValve torque\nV-12\n45\nV-13\n60\n2027"


def test_docx_table_cells_that_are_only_numbers_are_kept(pipeline):
    import io

    import docx

    document = docx.Document()
    table = document.add_table(rows=3, cols=2)
    for row, (name, torque) in enumerate([("Valve", "Torque"), ("V-12", "45"), ("V-13", "60")]):
        table.cell(row, 0).text = name
        table.cell(row, 1).text = torque
    buffer = io.BytesIO()
    document.save(buffer)
    pipeline.ingest("torque.docx", buffer.getvalue())
    stored = _stored_chunks(pipeline, "valve V-13 torque")[None]
    assert "V-12" in stored and "45" in stored and "V-13" in stored and "60" in stored


def test_a_page_number_at_the_foot_of_a_pdf_page_is_dropped_but_numbers_inside_the_page_stay(pipeline):
    pipeline.ingest("manual.pdf", make_pdf(["Valve torque\nV-12\n45\nV-13\n60\n1", "Second page text.\n2"]))
    assert _stored_chunks(pipeline, "valve torque second page") == {
        1: "Valve torque\nV-12\n45\nV-13\n60",
        2: "Second page text.",
    }
