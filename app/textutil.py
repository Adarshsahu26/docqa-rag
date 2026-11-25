"""Tiny tokenizer shared by the offline embedder and the extractive answerer."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about after all also an and any are as at be been but by can could did do does
    for from had has have how i if in into is it its of on or our so than that the their
    then there these they this to was we were what when where which who why will with
    would you your""".split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Very light plural stripping so 'bearings' matches 'bearing'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case, split into alphanumeric words, drop stop-words, strip plurals."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
