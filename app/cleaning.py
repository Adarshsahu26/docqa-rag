"""Text cleaning: turn raw extractor output into tidy prose before chunking."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_AT_LINE_END = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_PAGE_NUMBER_LINE = re.compile(
    r"^(?:[-–—]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?(?:\s*[-–—])?$",
    re.IGNORECASE,
)
_INLINE_SPACE = re.compile(r"[ \t ]+")
_BLANK_LINE_RUN = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")
_SENTENCE_END_CHARS = ".!?:;"


def clean_text(text: str, strip_page_numbers: bool = True) -> str:
    """Normalise one block of extracted text.

    Steps: Unicode NFKC -> unix newlines -> drop control characters -> trim every
    line -> drop a page number on the first or last line -> re-join words hyphenated
    across a line break -> re-join lines that were only wrapped for layout -> collapse
    blank lines.

    A page number sits alone at the top or the bottom of a page, so only the first and
    the last line are checked. A number on a line of its own anywhere else (a table cell,
    a year, a list number) is content and stays. Pass strip_page_numbers=False for formats
    without pages (DOCX, TXT, MD), where such a line is never a page number.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    lines = [_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    if strip_page_numbers:
        lines = _drop_edge_page_numbers(lines)
    text = _HYPHEN_AT_LINE_END.sub("", "\n".join(lines))
    text = _unwrap(text)
    return _BLANK_LINE_RUN.sub("\n\n", text).strip()


def _drop_edge_page_numbers(lines: list[str]) -> list[str]:
    """Remove the first and the last non-empty line when it is nothing but a page number."""
    filled = [i for i, line in enumerate(lines) if line]
    if not filled:
        return lines
    drop = {i for i in (filled[0], filled[-1]) if _PAGE_NUMBER_LINE.match(lines[i])}
    return [line for i, line in enumerate(lines) if i not in drop]


def _unwrap(text: str) -> str:
    """Join lines that were wrapped for layout rather than ended on purpose.

    A line continues the previous one when that previous line has no closing
    punctuation and this line starts with a lower-case letter.
    """
    out: list[str] = []
    for line in text.split("\n"):
        continues = out and out[-1] and line and line[0].islower() and out[-1][-1] not in _SENTENCE_END_CHARS
        if continues:
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


def drop_repeated_lines(
    pages: list[str], min_pages: int = 4, min_share: float = 0.6, max_len: int = 100
) -> list[str]:
    """Remove running headers/footers: short lines that repeat on most pages.

    Digits are ignored when comparing, so 'Manual v2 - Page 3' and '... Page 4'
    count as the same line. Documents shorter than `min_pages` are returned
    unchanged because there is too little evidence to call anything boilerplate.
    """
    if len(pages) < min_pages:
        return pages

    def key(line: str) -> str:
        return _DIGITS.sub("#", line.strip())

    seen: Counter[str] = Counter()
    for page in pages:
        seen.update({key(line) for line in page.splitlines() if line.strip()})
    boilerplate = {k for k, n in seen.items() if n / len(pages) >= min_share and len(k) <= max_len}
    return ["\n".join(line for line in page.splitlines() if key(line) not in boilerplate) for page in pages]
