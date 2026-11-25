"""Parsing real PDF, DOCX, TXT and MD bytes (the PDF and DOCX files are generated on the fly)."""
import codecs

import pytest

from app.errors import DocumentParseError, UnsupportedFileType
from app.parsing import parse_document
from tests.helpers import make_docx, make_pdf


@pytest.mark.parametrize("name", ["notes.txt", "README.md", "SHOUTING.TXT"])
def test_text_files_become_a_single_page_without_a_number(name):
    pages = parse_document(name, b"Hello world.")
    assert [(page.number, page.text) for page in pages] == [(None, "Hello world.")]


def test_a_utf8_byte_order_mark_is_removed():
    assert parse_document("a.txt", b"\xef\xbb\xbfHello")[0].text == "Hello"


def test_text_that_is_not_utf8_is_read_as_windows_1252():
    # What older Windows programs save as "ANSI": a lone \xe9 is invalid UTF-8 but is an e-acute in Windows-1252.
    original = "Das Ventil muss geprüft werden. Café – “quoted”."
    assert parse_document("a.txt", original.encode("cp1252"))[0].text == original


@pytest.mark.parametrize("mark, codec", [(codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")])
def test_utf16_text_with_a_byte_order_mark_is_decoded(mark, codec):
    original = "naïve café – valve CT-88"
    assert parse_document("a.txt", mark + original.encode(codec))[0].text == original


def test_valid_utf8_is_never_misread_as_windows_1252():
    original = "café – über ✓"  # in Windows-1252 these bytes would read as mojibake
    assert parse_document("a.md", original.encode("utf-8"))[0].text == original


def test_bytes_that_no_encoding_defines_are_replaced_instead_of_crashing():
    text = parse_document("a.txt", b"before \x81 after")[0].text  # 0x81 is undefined in Windows-1252
    assert text.startswith("before")
    assert text.endswith("after")


def test_pdf_pages_are_numbered_from_one():
    pages = parse_document("manual.pdf", make_pdf(["First page text.", "Second page text."]))
    assert [page.number for page in pages] == [1, 2]
    assert "First page text." in pages[0].text
    assert "Second page text." in pages[1].text


def test_a_blank_pdf_page_keeps_its_number_and_has_empty_text():
    pages = parse_document("manual.pdf", make_pdf(["Some text.", ""]))
    assert [page.number for page in pages] == [1, 2]
    assert pages[1].text.strip() == ""


def test_docx_paragraphs_are_extracted_as_one_unnumbered_page():
    pages = parse_document("report.docx", make_docx(["Alpha paragraph.", "Beta paragraph."]))
    assert len(pages) == 1
    assert pages[0].number is None
    assert "Alpha paragraph." in pages[0].text
    assert "Beta paragraph." in pages[0].text


@pytest.mark.parametrize("name", ["virus.exe", "archive.zip", "picture.png", "no_extension", ""])
def test_unsupported_file_types_are_rejected(name):
    with pytest.raises(UnsupportedFileType):
        parse_document(name, b"data")


def test_a_corrupt_pdf_raises_a_parse_error():
    with pytest.raises(DocumentParseError):
        parse_document("broken.pdf", b"this is definitely not a pdf")


def test_a_corrupt_docx_raises_a_parse_error():
    with pytest.raises(DocumentParseError):
        parse_document("broken.docx", b"this is definitely not a zip file")
