"""Offline keyword extraction: TF-IDF over a document's own chunks, with phrases.

A term is a good keyword for a chunk when it is frequent *there* but rare in the other chunks of
the same document. Two-word phrases ("refund policy") are preferred over their parts when they
repeat. No model, no corpus, deterministic: the same chunks always give the same keywords.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.ingest.enrichment.language import ALL_STOPWORDS

_TOKEN = re.compile(r"[^\W\d_][\w'’-]*", re.UNICODE)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
MIN_LEN = 3


def _terms(text: str) -> tuple[list[str], list[str]]:
    """Unigrams and adjacent-word bigrams of content words, lowercased."""
    tokens = [t.lower().strip("'’-") for t in _TOKEN.findall(text)]
    content = [t if (len(t) >= MIN_LEN and t not in ALL_STOPWORDS) else "" for t in tokens]
    unigrams = [t for t in content if t]
    bigrams = [
        f"{a} {b}" for a, b in zip(content, content[1:], strict=False) if a and b
    ]  # a stopword in between breaks the phrase
    return unigrams, bigrams


def extract_keywords(chunks: list[str], k: int = 8) -> list[list[str]]:
    """Top `k` keywords for every chunk (same order). CJK text has no spaces to split on, so
    it yields no keywords here rather than bad ones."""
    n = len(chunks)
    analysed = []
    df: Counter[str] = Counter()
    for text in chunks:
        if len(_CJK.findall(text)) > len(text) * 0.3:
            analysed.append(None)
            continue
        uni, bi = _terms(text)
        counts = Counter(uni) + Counter({b: c for b, c in Counter(bi).items() if c >= 2})
        analysed.append(counts)
        df.update(counts.keys())

    result: list[list[str]] = []
    for counts in analysed:
        if not counts:
            result.append([])
            continue
        scored = []
        for term, tf in counts.items():
            idf = math.log((1 + n) / (1 + df[term])) + 1.0  # smoothed; never zero
            phrase_bonus = 1.5 if " " in term else 1.0
            scored.append((tf * idf * phrase_bonus, term))
        scored.sort(key=lambda x: (-x[0], x[1]))
        picked: list[str] = []
        for _, term in scored:
            if len(picked) == k:
                break
            parts = term.split()
            if len(parts) == 1 and any(term in p.split() for p in picked if " " in p):
                continue  # already covered by a chosen phrase
            picked.append(term)
        result.append(picked)
    return result
