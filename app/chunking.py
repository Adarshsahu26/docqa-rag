"""Sentence-aware chunking with word-level overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass

# A sentence ends at . ! ? followed by whitespace and something that starts a new sentence.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class _Unit:
    text: str
    words: int
    new_line: bool  # True if this unit started a new line in the source text


def _units(text: str, max_words: int) -> list[_Unit]:
    """Cut text into sentences; every newline is also a boundary (headings, bullets)."""
    units: list[_Unit] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for j, sentence in enumerate(_SENTENCE_BREAK.split(line)):
            words = sentence.split()
            for k in range(0, len(words), max_words):  # hard-split over-long sentences
                piece = words[k : k + max_words]
                units.append(_Unit(" ".join(piece), len(piece), new_line=(j == 0 and k == 0)))
    return units


def split_sentences(text: str) -> list[str]:
    """Public helper: the sentences of `text`, in order."""
    return [unit.text for unit in _units(text, max_words=10**9)]


def _join(units: list[_Unit]) -> str:
    parts: list[str] = []
    for i, unit in enumerate(units):
        if i:
            parts.append("\n" if unit.new_line else " ")
        parts.append(unit.text)
    return "".join(parts)


def _overlap_tail(units: list[_Unit], overlap: int) -> list[_Unit]:
    """Longest run of whole units at the end of `units` that fits in `overlap` words."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.words > overlap:
            break
        tail.insert(0, unit)
        total += unit.words
    return tail


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Pack sentences into chunks of at most `chunk_size` words.

    Neighbouring chunks share up to `overlap` words (whole sentences), so a fact
    that straddles a boundary still appears intact in at least one chunk.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[_Unit] = []
    current_words = 0
    for unit in _units(text, max_words=chunk_size):
        if current and current_words + unit.words > chunk_size:
            chunks.append(_join(current))
            current = _overlap_tail(current, overlap)
            current_words = sum(u.words for u in current)
            if current_words + unit.words > chunk_size:  # overlap would break the size limit
                current, current_words = [], 0
        current.append(unit)
        current_words += unit.words
    if current:
        chunks.append(_join(current))
    return chunks
