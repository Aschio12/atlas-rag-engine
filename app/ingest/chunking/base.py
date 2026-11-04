"""Shared building blocks for chunkers.

Every chunker works with **spans** `(start, end)` of the parsed text and emits a `ChunkDraft`
whose `text` is exactly `parsed.text[start:end]`. Because chunks are never rebuilt from pieces,
offsets cannot drift, and a citation can always highlight the precise passage.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.ingest.chunking.config import ChunkingConfig
from app.ingest.parsers import Heading, ParsedDocument

Span = tuple[int, int]

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def estimate_tokens(text: str) -> int:
    """A model-independent token estimate: ~4 characters per token for alphabetic text, one
    token per CJK character, and never less than 1.3 tokens per word. Good enough to size
    chunks; exact counts depend on the embedding model's tokenizer."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    rest = len(text) - cjk
    by_chars = cjk + math.ceil(rest / 4)
    by_words = round(len(text.split()) * 1.3)
    return max(by_chars, by_words, 1)


@dataclass
class ChunkDraft:
    text: str
    char_start: int
    char_end: int
    page: int | None
    heading_path: list[str]
    token_count: int
    kind: str = "chunk"  # "chunk" | "parent" | "child"
    parent: int | None = None  # index of the parent draft, for children
    context: str | None = None  # contextual chunking: situating sentence(s)
    meta: dict = field(default_factory=dict)

    @property
    def indexable(self) -> bool:
        """Parents exist to give context back at answer time; only the rest are searched."""
        return self.kind != "parent"

    @property
    def embed_text(self) -> str:
        """What gets embedded: the situating context (if any) followed by the passage."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text


class HeadingIndex:
    """Answers 'which section is this offset in?' with the breadcrumb of headings above it."""

    def __init__(self, headings: list[Heading]):
        self._starts: list[int] = []
        self._paths: list[list[str]] = []
        stack: list[Heading] = []
        for h in sorted(headings, key=lambda h: h.char_start):
            while stack and stack[-1].level >= h.level:
                stack.pop()
            stack.append(h)
            self._starts.append(h.char_start)
            self._paths.append([x.text for x in stack])

    def path_at(self, offset: int) -> list[str]:
        i = bisect.bisect_right(self._starts, offset) - 1
        return list(self._paths[i]) if i >= 0 else []


def trim(text: str, start: int, end: int) -> Span | None:
    """Shrink a span to exclude surrounding whitespace; None if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def make_draft(
    parsed: ParsedDocument, headings: HeadingIndex, start: int, end: int, **kw
) -> ChunkDraft | None:
    span = trim(parsed.text, start, end)
    if span is None:
        return None
    s, e = span
    piece = parsed.text[s:e]
    return ChunkDraft(
        text=piece,
        char_start=s,
        char_end=e,
        page=parsed.page_at(s),
        heading_path=headings.path_at(s),
        token_count=estimate_tokens(piece),
        **kw,
    )


# --- sentences ------------------------------------------------------------------------------

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig", "no",
    "inc", "ltd", "co", "approx", "dept", "est",
}  # fmt: skip
_SENTENCE_END = re.compile(r"""[.!?…。！？]+["'”’)\]]*(?=\s|$)""")


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    """Split `text[start:end]` into sentence-like units (also at every line break, so list
    items, table rows and CSV records stay separate). Spans cover the text without overlap and
    exclude only whitespace between units."""
    end = len(text) if end is None else end
    spans: list[Span] = []
    for line_match in re.finditer(r"[^\n]+", text[start:end]):
        line_start = start + line_match.start()
        line = line_match.group()
        cursor = 0
        for m in _SENTENCE_END.finditer(line):
            if m.group()[0] == ".":
                before = line[cursor : m.start()]
                word = re.search(r"([A-Za-z.]+)$", before)
                if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
                    continue  # "Dr. Smith", "e.g. this"
                if re.fullmatch(r"\s*(\d{1,3}|[A-Za-z])", before):
                    continue  # a list marker such as "1." or "a."
            span = trim(text, line_start + cursor, line_start + m.end())
            if span:
                spans.append(span)
            cursor = m.end()
        tail = trim(text, line_start + cursor, line_start + len(line))
        if tail:
            spans.append(tail)
    return spans


class Chunker(Protocol):
    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        """Split a parsed document. Chunks are returned in document order."""
