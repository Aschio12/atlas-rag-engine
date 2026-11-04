"""Invariants every chunker must satisfy, shared by the chunking tests."""

from __future__ import annotations

from app.ingest.chunking import ChunkDraft, ChunkingConfig, estimate_tokens
from app.ingest.chunking.base import HeadingIndex
from app.ingest.parsers import ParsedDocument


def check_chunks(
    parsed: ParsedDocument,
    drafts: list[ChunkDraft],
    cfg: ChunkingConfig,
    *,
    cover: bool = True,
    max_tokens: int | None = None,
) -> None:
    text = parsed.text
    headings = HeadingIndex(parsed.headings)
    limit = max_tokens or cfg.max_tokens
    previous = (-1, -1)
    for d in drafts:
        # exact slice: offsets can be trusted for citation highlighting
        assert text[d.char_start : d.char_end] == d.text
        assert d.text == d.text.strip() and d.text
        assert d.token_count == estimate_tokens(d.text)
        assert d.token_count <= limit, (d.token_count, limit)
        assert d.page == parsed.page_at(d.char_start)
        assert d.heading_path == headings.path_at(d.char_start)
        # document order, always moving forward
        assert (d.char_start, d.char_end) > previous
        assert d.char_end > previous[1]
        previous = (d.char_start, d.char_end)
    if cover:
        covered = bytearray(len(text))
        for d in drafts:
            covered[d.char_start : d.char_end] = b"\x01" * (d.char_end - d.char_start)
        lost = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
        assert not lost, f"text lost at offsets {lost[:10]}"


def overlap_chars(drafts: list[ChunkDraft]) -> list[int]:
    return [max(a.char_end - b.char_start, 0) for a, b in zip(drafts, drafts[1:], strict=False)]
