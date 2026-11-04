"""Fixed-size chunks with overlap (2.4.1): the baseline every other strategy is compared to."""

from __future__ import annotations

from app.ingest.chunking.base import ChunkDraft, HeadingIndex, estimate_tokens, make_draft
from app.ingest.chunking.config import ChunkingConfig
from app.ingest.parsers import ParsedDocument

SNAP_WINDOW = 0.2  # how far back (share of the window) a cut may move to land on whitespace


class FixedChunker:
    """Sliding window over the text. Cuts are nudged back to the nearest whitespace so words
    are never split, but sentences and sections are ignored on purpose: that is what makes it
    a clean baseline."""

    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        text = parsed.text
        n = len(text)
        if not n:
            return []
        # Tokens per character for *this* document, so CJK and prose get comparable windows.
        density = max(estimate_tokens(text) / n, 1e-3)
        window = max(int(cfg.target_tokens / density), 8)
        overlap = int(cfg.overlap_tokens / density)
        headings = HeadingIndex(parsed.headings)

        drafts: list[ChunkDraft] = []
        pos = 0
        while pos < n:
            end = min(pos + window, n)
            if end < n:
                end = _snap_back(text, pos, end, int(window * SNAP_WINDOW))
            if draft := make_draft(parsed, headings, pos, end):
                drafts.append(draft)
            if end >= n:
                break
            nxt = _snap_forward(text, max(end - overlap, pos + 1), end)
            pos = nxt if nxt > pos else end
        return drafts


def _snap_back(text: str, pos: int, end: int, slack: int) -> int:
    """Move `end` back to just after whitespace, within `slack` characters, if possible."""
    for i in range(end, max(end - slack, pos + 1), -1):
        if text[i - 1].isspace() or text[i].isspace():
            return i
    return end


def _snap_forward(text: str, start: int, limit: int) -> int:
    """Move `start` forward to the beginning of a word (never past `limit`)."""
    i = start
    while 0 < i < limit and not text[i - 1].isspace():
        i += 1
    return min(i, limit)
