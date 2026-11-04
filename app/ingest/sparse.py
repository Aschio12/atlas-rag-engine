"""Sparse (lexical, BM25-style) vectors.

A chunk becomes a map `term id -> weight`. The weight is BM25's term-frequency half,
`tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))`; the vector store supplies the
other half (inverse document frequency) at query time, which is exactly how Qdrant's `idf`
modifier works. The same analyzer is used for documents and queries, so they always agree.

Terms are lowercase words with accents folded, a very light English stemmer (so "refunds" and
"refunding" meet "refund"), and character bigrams for CJK text, which has no spaces to split on.
Term ids are CRC32 hashes, which fit Qdrant's unsigned 32-bit sparse indices; collisions
among a few hundred thousand distinct terms are rare and only blur, never break, scoring.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

from app.providers.base import SparseVector

K1 = 1.2
B = 0.75
AVG_LENGTH = 120.0  # typical chunk length in terms; a constant keeps documents independent
MAX_TERM_LENGTH = 40

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]+")


def _fold(text: str) -> str:
    """Lowercase and strip accents (cafe == café)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def stem(word: str) -> str:
    """A deliberately light English stemmer; non-English or short words pass through."""
    if len(word) <= 4 or not word.isascii() or not word.isalpha():
        return word
    for suffix, replacement in (("sses", "ss"), ("ies", "y")):
        if word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    out: list[str] = []
    for token in _WORD.findall(_fold(text)):
        if _CJK.fullmatch(token):
            chars = list(token)
            out.extend(
                chars
                if len(chars) == 1
                else (a + b for a, b in zip(chars, chars[1:], strict=False))
            )
            continue
        if len(token) > MAX_TERM_LENGTH or (len(token) < 2 and not token.isdigit()):
            continue
        out.append(stem(token))
    return out


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0xFFFFFFFF


def _vector(weights: dict[int, float]) -> SparseVector:
    ordered = sorted(weights)
    return SparseVector(ordered, [round(weights[i], 5) for i in ordered])


def encode_document(text: str) -> SparseVector:
    """BM25 term-frequency weights for a chunk (or a hypothetical question)."""
    counts = Counter(terms(text))
    length = sum(counts.values())
    weights: dict[int, float] = {}
    for term, tf in counts.items():
        w = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / AVG_LENGTH))
        idx = term_id(term)
        weights[idx] = weights.get(idx, 0.0) + w  # hash collisions add up
    return _vector(weights)


def encode_query(text: str) -> SparseVector:
    """Query terms, each weighted once: repeating a word does not make it more important."""
    return _vector({term_id(t): 1.0 for t in set(terms(text))})
