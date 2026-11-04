"""Structure-aware recursive chunking (2.4.2): the default strategy.

Text is split along its natural seams, largest first: sections (headings), paragraphs, lines,
sentences, words, and only as a last resort characters. Neighbouring pieces are then merged
back together up to the target size. Consequences: chunks end at paragraph or sentence
boundaries, a heading stays with the text it introduces, a section break is respected unless the
chunk would be tiny, and overlap never straddles a section.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from app.ingest.chunking.base import (
    ChunkDraft,
    HeadingIndex,
    Span,
    estimate_tokens,
    make_draft,
    sentence_spans,
    trim,
)
from app.ingest.chunking.config import ChunkingConfig
from app.ingest.parsers import ParsedDocument

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_WORD = re.compile(r"\S+")


def _paragraphs(text: str, lo: int, hi: int) -> list[Span]:
    spans, cursor = [], lo
    for m in _PARAGRAPH_BREAK.finditer(text, lo, hi):
        if s := trim(text, cursor, m.start()):
            spans.append(s)
        cursor = m.end()
    if s := trim(text, cursor, hi):
        spans.append(s)
    return spans


def _lines(text: str, lo: int, hi: int) -> list[Span]:
    spans, cursor = [], lo
    for m in re.finditer(r"\n", text[lo:hi]):
        if s := trim(text, cursor, lo + m.start()):
            spans.append(s)
        cursor = lo + m.end()
    if s := trim(text, cursor, hi):
        spans.append(s)
    return spans


def _words(text: str, lo: int, hi: int) -> list[Span]:
    return [(lo + m.start(), lo + m.end()) for m in _WORD.finditer(text[lo:hi])]


SPLITTERS: list[Callable[[str, int, int], list[Span]]] = [
    _paragraphs,
    _lines,
    sentence_spans,
    _words,
]


def _hard_split(text: str, lo: int, hi: int, target: int) -> list[Span]:
    """Last resort for a single 'word' bigger than a chunk (a long URL, base64, minified code)."""
    density = max(estimate_tokens(text[lo:hi]) / max(hi - lo, 1), 1e-3)
    step = max(int(target / density), 1)
    return [(i, min(i + step, hi)) for i in range(lo, hi, step)]


def split_span(text: str, lo: int, hi: int, target: int, level: int = 0) -> list[Span]:
    """Break text[lo:hi] into pieces of at most `target` tokens, using the coarsest separator
    that works. Pieces are returned in order and are exact, trimmed slices."""
    whole = trim(text, lo, hi)
    if whole is None:
        return []
    if estimate_tokens(text[whole[0] : whole[1]]) <= target:
        return [whole]
    while level < len(SPLITTERS):
        pieces = SPLITTERS[level](text, *whole)
        if len(pieces) > 1:
            out: list[Span] = []
            for p in pieces:
                out.extend(split_span(text, p[0], p[1], target, level + 1))
            return out
        level += 1
    return _hard_split(text, *whole, target)


def recursive_spans(
    parsed: ParsedDocument,
    lo: int,
    hi: int,
    *,
    target: int,
    overlap: int,
    min_tokens: int,
    max_tokens: int,
) -> list[Span]:
    """Chunk spans for text[lo:hi]; see the module docstring for the rules."""
    text = parsed.text
    heading_starts = sorted({h.char_start for h in parsed.headings if lo <= h.char_start < hi})
    heading_line_end = {
        s: (text.find("\n", s) if text.find("\n", s) != -1 else len(text)) for s in heading_starts
    }

    # 1. pieces, remembering which ones begin a section
    bounds = [lo, *[s for s in heading_starts if s > lo], hi]
    pieces: list[tuple[Span, bool]] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        for i, span in enumerate(split_span(text, a, b, target)):
            pieces.append((span, i == 0 and a in heading_starts))

    def tokens(a: int, b: int) -> int:
        return estimate_tokens(text[a:b])

    def is_heading_only(span: Span) -> bool:
        return span[0] in heading_line_end and span[1] <= heading_line_end[span[0]]

    def overlap_seed(a: int, b: int) -> int | None:
        """Start of the trailing sentences of text[a:b] that fit in `overlap` tokens."""
        if not overlap:
            return None
        start = None
        for s, _ in reversed(sentence_spans(text, a, b)):
            if tokens(s, b) > overlap:
                break
            start = s
        return start if start is not None and start > a else None

    # 2. merge greedily
    chunks: list[Span] = []
    cur: list[tuple[Span, bool]] = []
    for piece in pieces:
        span, starts_section = piece
        if cur:
            cur_tokens = tokens(cur[0][0][0], cur[-1][0][1])
            # A heading alone may exceed the target so it is never stranded from its text.
            limit = max_tokens if all(is_heading_only(c[0]) for c in cur) else target
            too_big = tokens(cur[0][0][0], span[1]) > limit
            section_break = starts_section and cur_tokens >= min_tokens
            if too_big or section_break:
                carry: list[tuple[Span, bool]] = []
                while len(cur) > 1 and is_heading_only(cur[-1][0]):
                    carry.insert(0, cur.pop())  # never leave a heading dangling at a chunk's end
                first, last = cur[0][0][0], cur[-1][0][1]
                chunks.append((first, last))
                cur = carry
                if not carry and not starts_section:
                    seed = overlap_seed(first, last)
                    if seed is not None and tokens(seed, span[1]) <= max_tokens:
                        cur = [((seed, last), False)]
        cur.append(piece)
    if cur:
        chunks.append((cur[0][0][0], cur[-1][0][1]))
    return chunks


class RecursiveChunker:
    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        headings = HeadingIndex(parsed.headings)
        spans = recursive_spans(
            parsed,
            0,
            len(parsed.text),
            target=cfg.target_tokens,
            overlap=cfg.overlap_tokens,
            min_tokens=cfg.min_tokens,
            max_tokens=cfg.max_tokens,
        )
        drafts = (make_draft(parsed, headings, s, e) for s, e in spans)
        return [d for d in drafts if d is not None]
