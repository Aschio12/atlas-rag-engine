"""Parent-child chunking (2.4.4): search small, answer with big.

Small child chunks are what gets embedded and searched, because short passages match a query
precisely. Each child points at a large parent chunk, which is what the generator receives, so
the answer sees the surrounding context instead of an isolated fragment. Parents are stored but
never indexed (`indexable` is False for them).

Output order is `parent, child, child, ..., parent, child, ...` and `child.parent` is the index
of its parent in that list.
"""

from __future__ import annotations

from app.ingest.chunking.base import ChunkDraft, HeadingIndex, make_draft
from app.ingest.chunking.config import ChunkingConfig
from app.ingest.chunking.recursive import recursive_spans
from app.ingest.parsers import ParsedDocument

PARENT_SLACK = 1.25  # a parent may exceed its target by this factor to avoid tiny leftovers


class ParentChildChunker:
    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        headings = HeadingIndex(parsed.headings)
        parent_spans = recursive_spans(
            parsed,
            0,
            len(parsed.text),
            target=cfg.parent_tokens,
            overlap=0,
            min_tokens=max(cfg.parent_tokens // 4, 16),
            max_tokens=int(cfg.parent_tokens * PARENT_SLACK),
        )
        child_max = max(cfg.child_tokens, min(cfg.max_tokens, cfg.child_tokens * 2))
        drafts: list[ChunkDraft] = []
        for start, end in parent_spans:
            parent = make_draft(parsed, headings, start, end, kind="parent")
            if parent is None:
                continue
            parent_index = len(drafts)
            drafts.append(parent)
            child_spans = recursive_spans(
                parsed,
                parent.char_start,
                parent.char_end,
                target=cfg.child_tokens,
                overlap=min(cfg.overlap_tokens, cfg.child_tokens // 2),
                min_tokens=max(cfg.child_tokens // 4, 16),
                max_tokens=child_max,
            )
            for s, e in child_spans:
                child = make_draft(parsed, headings, s, e, kind="child", parent=parent_index)
                if child is not None:
                    drafts.append(child)
        return drafts
