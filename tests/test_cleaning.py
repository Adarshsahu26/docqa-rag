from app.cleaning import clean_text, drop_repeated_lines


def test_normalises_unicode_ligatures_and_non_breaking_spaces():
    assert clean_text("The ﬁnal report") == "The final report"


def test_removes_control_characters():
    assert clean_text("Alpha\x00beta\x07gamma") == "Alphabetagamma"


def test_converts_windows_and_old_mac_line_endings():
    assert clean_text("One.\r\nTwo.\rThree.") == "One.\nTwo.\nThree."


def test_drops_a_page_number_on_the_first_or_the_last_line_of_a_page():
    body = "First paragraph ends here.\nSecond paragraph."
    assert clean_text(f"12\n{body}") == body
    assert clean_text(f"{body}\nPage 3 of 10") == body
    assert clean_text("Only text.\n- 4 -\n\n") == "Only text."
    assert clean_text("7\nOnly text.\n8") == "Only text."


def test_keeps_a_number_that_stands_alone_inside_the_page():
    table = "Valve\nTorque (Nm)\nV-12\n45\nV-13\n60\nEnd of the table."
    assert clean_text(table) == table
    assert clean_text("Budget\n2026\nApproved by the board.") == "Budget\n2026\nApproved by the board."


def test_page_numbers_are_left_alone_for_formats_without_pages():
    text = "2026\nBudget\n45"
    assert clean_text(text, strip_page_numbers=False) == text
    assert clean_text(text) == "Budget"


def test_keeps_numbers_that_belong_to_a_sentence():
    assert clean_text("The limit is 12 kV.") == "The limit is 12 kV."


def test_rejoins_words_hyphenated_across_a_line_break():
    assert clean_text("routine mainte-\nnance is done") == "routine maintenance is done"


def test_keeps_genuine_hyphens():
    assert clean_text("a state-of-the-art system") == "a state-of-the-art system"


def test_unwraps_lines_that_were_only_wrapped_for_layout():
    wrapped = "The transformer oil must be sampled every\ntwelve months by a certified\ntechnician."
    assert clean_text(wrapped) == "The transformer oil must be sampled every twelve months by a certified technician."


def test_keeps_headings_and_finished_sentences_on_their_own_lines():
    text = "Inspection schedule\nVisual checks happen monthly.\nThen the oil is sampled."
    assert clean_text(text) == text


def test_collapses_long_runs_of_blank_lines():
    assert clean_text("One.\n\n\n\n\nTwo.") == "One.\n\nTwo."


def test_empty_and_whitespace_only_input():
    assert clean_text("") == ""
    assert clean_text("  \n\t \n") == ""


# ---- running headers and footers ------------------------------------------------------------
TOPICS = ["lubrication", "alignment", "insulation", "calibration", "ventilation"]


def _manual_pages(n: int = 5) -> list[str]:
    return [
        f"ACME Maintenance Manual v2\nThe {TOPICS[i]} section covers {TOPICS[i]} checks.\nPage {i + 1} of {n}"
        for i in range(n)
    ]


def test_removes_running_headers_and_footers_even_though_the_page_number_changes():
    cleaned = drop_repeated_lines(_manual_pages())
    for i, page in enumerate(cleaned):
        assert "ACME" not in page
        assert "Page " not in page
        assert f"The {TOPICS[i]} section covers {TOPICS[i]} checks." in page


def test_short_documents_are_left_alone():
    pages = _manual_pages(3)
    assert drop_repeated_lines(pages) == pages


def test_a_line_that_repeats_on_only_a_few_pages_is_kept():
    pages = _manual_pages(5)
    pages[0] += "\nNote: special instruction"
    pages[1] += "\nNote: special instruction"  # 2 of 5 pages = 40%, below the 60% threshold
    cleaned = drop_repeated_lines(pages)
    assert "special instruction" in cleaned[0]
    assert "special instruction" in cleaned[1]
