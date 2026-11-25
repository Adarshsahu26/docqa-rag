import pytest

from app.chunking import chunk_text, split_sentences


def n_words(text: str) -> int:
    return len(text.split())


# Forty sentences of exactly ten words each.
SENTENCES = [f"Sentence number {i} talks about pump {i} and valve {i}." for i in range(40)]
TEXT = " ".join(SENTENCES)


def test_short_text_is_a_single_chunk():
    assert chunk_text("One short sentence.", chunk_size=50, overlap=5) == ["One short sentence."]


@pytest.mark.parametrize("size", [15, 30, 55, 120])
def test_chunks_never_exceed_the_size_limit(size):
    chunks = chunk_text(TEXT, chunk_size=size, overlap=size // 5)
    assert len(chunks) > 1
    assert all(n_words(chunk) <= size for chunk in chunks)


def test_size_limit_holds_for_sentences_of_very_different_lengths():
    text = " ".join("word " * length + "end." for length in (3, 25, 7, 40, 2, 18, 11, 33, 5))
    for chunk in chunk_text(text, chunk_size=45, overlap=12):
        assert n_words(chunk) <= 45


def test_no_sentence_is_lost():
    joined = " ".join(chunk_text(TEXT, chunk_size=30, overlap=12))
    assert all(sentence in joined for sentence in SENTENCES)


def test_neighbouring_chunks_share_a_whole_sentence_when_overlap_is_allowed():
    chunks = chunk_text(TEXT, chunk_size=30, overlap=12)
    for before, after in zip(chunks, chunks[1:], strict=False):
        assert after.startswith(split_sentences(before)[-1])


def test_zero_overlap_repeats_nothing():
    chunks = chunk_text(TEXT, chunk_size=30, overlap=0)
    assert sum(n_words(chunk) for chunk in chunks) == n_words(TEXT)


def test_a_sentence_longer_than_the_limit_is_split_into_pieces():
    chunks = chunk_text("word " * 500, chunk_size=100, overlap=10)
    assert len(chunks) == 5
    assert all(n_words(chunk) == 100 for chunk in chunks)


def test_line_breaks_are_kept_so_headings_stay_on_their_own_line():
    text = "Purpose\nThis manual covers pumps.\nScope\nIt applies to all staff."
    assert chunk_text(text, chunk_size=100, overlap=10) == [text]


@pytest.mark.parametrize("text", ["", "   ", "\n\n \n"])
def test_blank_text_gives_no_chunks(text):
    assert chunk_text(text) == []


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_sizes_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("Some text here.", chunk_size=size, overlap=overlap)


def test_split_sentences_handles_decimals_and_line_breaks():
    assert split_sentences("The limit is 4.2 MW. Use the backup. Done!") == [
        "The limit is 4.2 MW.",
        "Use the backup.",
        "Done!",
    ]
    assert split_sentences("Heading\nBody text.") == ["Heading", "Body text."]


def test_a_chunk_may_hold_exactly_the_size_limit():
    text = "One two three four five. Six seven eight nine ten."
    assert chunk_text(text, chunk_size=10, overlap=0) == [text]
    assert chunk_text(text, chunk_size=9, overlap=0) == ["One two three four five.", "Six seven eight nine ten."]


def test_overlap_is_dropped_when_it_would_push_a_chunk_over_the_size_limit():
    # Carrying the last 5-word sentence into the next chunk would make that chunk 13 words long.
    text = (
        "One two three four five. Six seven eight nine ten. "
        "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen."
    )
    chunks = chunk_text(text, chunk_size=10, overlap=5)
    assert chunks == [
        "One two three four five. Six seven eight nine ten.",
        "Eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen.",
    ]
