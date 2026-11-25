"""Read uploaded files into plain text, one entry per page when pages exist."""
from __future__ import annotations

import codecs
import io
from dataclasses import dataclass
from pathlib import PurePath

from .errors import DocumentParseError, UnsupportedFileType

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


@dataclass(frozen=True)
class Page:
    number: int | None  # 1-based for PDFs; None for formats without pages
    text: str


def parse_document(filename: str, data: bytes) -> list[Page]:
    """Extract text from PDF, DOCX, TXT or MD bytes."""
    ext = PurePath(filename).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(data)
    if ext == ".docx":
        return _parse_docx(data)
    if ext in (".txt", ".md"):
        return [Page(None, _decode_text(data))]
    raise UnsupportedFileType(
        f"Unsupported file type '{ext or filename}'. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def _decode_text(data: bytes) -> str:
    """Decode a text file the way people really save them.

    UTF-8 (with or without a byte order mark) is tried first, and UTF-16 is recognised by its mark (that is
    what Notepad's "Unicode" option writes). Anything else is read as Windows-1252, which Windows calls "ANSI":
    older text files from Windows are almost always in it, and reading them as UTF-8 would turn every accented
    letter into a replacement character.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")  # a few byte values are undefined in Windows-1252


def _parse_pdf(data: bytes) -> list[Page]:
    import pdfplumber  # imported lazily: only needed for PDFs

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return [Page(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]
    except Exception as exc:  # pdfminer raises many different exception types
        raise DocumentParseError("Could not read this PDF (corrupt or password-protected?).") from exc


def _parse_docx(data: bytes) -> list[Page]:
    import docx2txt  # imported lazily: only needed for DOCX

    try:
        return [Page(None, docx2txt.process(io.BytesIO(data)))]
    except Exception as exc:
        raise DocumentParseError("Could not read this DOCX file.") from exc
