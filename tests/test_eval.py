"""The retrieval evaluation: its helpers, the integrity of the question set, and a regression guard."""
import json

import pytest

from app.embeddings import HashingEmbedder
from app.rag import Hit
from evaluation.run_eval import (
    DEFAULT_CORPUS,
    DEFAULT_QUESTIONS,
    evaluate,
    first_hit_rank,
    format_table,
    load_questions,
    normalise,
)
from tests.helpers import rec


def hit(text: str, filename: str = "a.txt", score: float = 0.5) -> Hit:
    return Hit(rec("d", 0, text, filename), score)


# ---- helpers -----------------------------------------------------------------------------------
def test_normalise_ignores_case_and_whitespace_differences():
    assert normalise("  Oil   is\nsampled  ") == "oil is sampled"


def test_the_rank_is_that_of_the_first_chunk_containing_the_evidence():
    hits = [hit("nothing here"), hit("Oil is sampled every 12   months."), hit("oil is sampled every 12 months")]
    assert first_hit_rank(hits, "sampled every 12 months", "a.txt") == 2


def test_a_chunk_from_the_wrong_file_does_not_count():
    hits = [hit("sampled every 12 months", filename="other.txt"), hit("sampled every 12 months", filename="a.txt")]
    assert first_hit_rank(hits, "sampled every 12 months", "a.txt") == 2


def test_without_a_source_any_file_may_match():
    assert first_hit_rank([hit("sampled every 12 months", filename="whatever.txt")], "every 12 months", None) == 1


def test_evidence_that_is_never_retrieved_gives_no_rank():
    assert first_hit_rank([hit("something else")], "every 12 months", "a.txt") is None


def test_questions_must_have_a_question_and_evidence(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"question": "No evidence given?"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        load_questions(path)


def test_blank_lines_in_the_question_file_are_skipped(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('\n{"question": "Q?", "evidence": "e"}\n\n', encoding="utf-8")
    assert load_questions(path) == [{"question": "Q?", "evidence": "e"}]


# ---- the bundled question set --------------------------------------------------------------------
QUESTIONS = load_questions(DEFAULT_QUESTIONS)


def test_every_question_names_a_file_that_exists_and_quotes_it_exactly():
    """A typo in `evidence` would make a question impossible to answer and quietly lower every score."""
    for row in QUESTIONS:
        source = DEFAULT_CORPUS / row["source"]
        assert source.exists(), f"{row['source']} is missing"
        assert normalise(row["evidence"]) in normalise(source.read_text(encoding="utf-8")), row["question"]


def test_questions_are_unique_and_cover_every_document():
    questions = [row["question"] for row in QUESTIONS]
    assert len(set(questions)) == len(questions)
    assert {row["source"] for row in QUESTIONS} == {path.name for path in DEFAULT_CORPUS.glob("*.txt")}


# ---- running it --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def baseline():
    return evaluate(HashingEmbedder(), DEFAULT_CORPUS, QUESTIONS, chunk_size=120, overlap=18)


def test_the_offline_baseline_finds_the_evidence_in_the_top_five_for_nearly_every_question(baseline):
    # Measured at 1.00 when this test was written; the margin leaves room for small chunking changes.
    assert baseline.hit_at[5] >= 0.9
    assert baseline.hit_at[1] <= baseline.hit_at[3] <= baseline.hit_at[5]
    assert 0 < baseline.mrr <= 1


def test_the_result_records_how_it_was_produced(baseline):
    assert baseline.embedder == "hashing-4096"
    assert (baseline.chunk_size, baseline.overlap) == (120, 18)
    assert baseline.chunks > 0


def test_the_table_has_one_row_per_result_and_a_column_per_k(baseline):
    table = format_table([baseline, baseline], ks=(1, 3, 5))
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert all(column in lines[0] for column in ("hit@1", "hit@3", "hit@5", "MRR"))
    assert "hashing-4096" in lines[2]
