"""Semantic chunking (2.4.3): cut where the topic changes.

Sentences are embedded (each together with its neighbours, which smooths out single-sentence
noise), the distance between consecutive sentences is measured, and a cut is made wherever that
distance is unusually large (above the configured percentile). Size rules still apply: never
below `min_tokens` just because of a topic wobble, never above `max_tokens`, and a new section
(heading) starts a new chunk once the current one is big enough.

It costs one embedding call per sentence, so it is slower and dearer than the structural
strategies; the evaluation harness (Phase 6) measures whether it earns that.
"""

from __future__ import annotations

import math

from app.ingest.chunking.base import (
    ChunkDraft,
    HeadingIndex,
    Span,
    estimate_tokens,
    make_draft,
    sentence_spans,
)
from app.ingest.chunking.config import ChunkingConfig
from app.ingest.chunking.recursive import split_span
from app.ingest.parsers import ParsedDocument
from app.providers.base import Embedder

BATCH = 64
NEIGHBOURS = 1  # each sentence is embedded together with this many on each side


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 1.0


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0-100) without needing numpy."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


class SemanticChunker:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    async def chunk(self, parsed: ParsedDocument, cfg: ChunkingConfig) -> list[ChunkDraft]:
        text = parsed.text
        units = self._units(text, cfg)
        headings = HeadingIndex(parsed.headings)
        if not units:
            return []
        if estimate_tokens(text[units[0][0] : units[-1][1]]) <= cfg.target_tokens:
            return [d] if (d := make_draft(parsed, headings, units[0][0], units[-1][1])) else []

        distances = await self._distances(text, units)
        threshold = percentile(distances, cfg.breakpoint_percentile)
        section_starts = {h.char_start for h in parsed.headings}

        def tokens(a: int, b: int) -> int:
            return estimate_tokens(text[units[a][0] : units[b][1]])

        spans: list[Span] = []
        first = 0
        for i in range(len(units) - 1):
            size = tokens(first, i)
            topic_shift = distances[i] > threshold and size >= cfg.min_tokens
            new_section = units[i + 1][0] in section_starts and size >= cfg.min_tokens
            overflow = tokens(first, i + 1) > cfg.max_tokens
            if topic_shift or new_section or overflow:
                spans.append((units[first][0], units[i][1]))
                first = i + 1
        spans.append((units[first][0], units[-1][1]))
        spans = self._merge_small_tail(text, spans, cfg)
        drafts = (make_draft(parsed, headings, s, e) for s, e in spans)
        return [d for d in drafts if d is not None]

    @staticmethod
    def _units(text: str, cfg: ChunkingConfig) -> list[Span]:
        """Sentences, with any sentence bigger than a whole chunk broken down further."""
        units: list[Span] = []
        for s, e in sentence_spans(text):
            if estimate_tokens(text[s:e]) > cfg.max_tokens:
                units.extend(split_span(text, s, e, cfg.target_tokens, level=2))
            else:
                units.append((s, e))
        return units

    async def _distances(self, text: str, units: list[Span]) -> list[float]:
        n = len(units)
        windows = [
            text[units[max(i - NEIGHBOURS, 0)][0] : units[min(i + NEIGHBOURS, n - 1)][1]]
            for i in range(n)
        ]
        vectors: list[list[float]] = []
        for i in range(0, n, BATCH):
            vectors.extend(await self.embedder.embed(windows[i : i + BATCH]))
        if len(vectors) != n:
            raise ValueError("the embedder returned a different number of vectors than inputs")
        return [cosine_distance(vectors[i], vectors[i + 1]) for i in range(n - 1)]

    @staticmethod
    def _merge_small_tail(text: str, spans: list[Span], cfg: ChunkingConfig) -> list[Span]:
        """A trailing sliver is attached to the previous chunk if the result still fits."""
        if len(spans) >= 2:
            last = estimate_tokens(text[spans[-1][0] : spans[-1][1]])
            merged = estimate_tokens(text[spans[-2][0] : spans[-1][1]])
            if last < cfg.min_tokens and merged <= cfg.max_tokens:
                spans = [*spans[:-2], (spans[-2][0], spans[-1][1])]
        return spans
