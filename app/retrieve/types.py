"""What retrieval returns: ranked hits and a trace of how they were found.

A `Hit` is one passage with the score each stage gave it and the reasons it is here. The trace
records what every stage did, with a snapshot of the best candidates after each ranking stage and
the details of the passages they refer to, so a UI (the Retrieval Inspector) can replay a search
without touching the database. `to_dict()` is its stable, versioned, JSON-safe form.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Hit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    collection_id: uuid.UUID
    rank: int
    ordinal: int  # position of the chunk in its document
    score: float  # the score of the stage that produced the current order
    text: str
    title: str
    page: int | None
    char_start: int
    char_end: int
    heading_path: list[str]
    parent_id: uuid.UUID | None
    language: str | None
    source_uri: str | None
    matched: Literal["chunk", "question"]  # the passage itself, or a question generated for it
    question: str | None = None
    scores: dict[str, float] = field(default_factory=dict)  # per stage: {"dense": 0.83, ...}
    reasons: list[str] = field(default_factory=list)  # why this passage is in the result


TRACE_SCHEMA_VERSION = 1
SNAPSHOT_SIZE = 20  # candidates listed per ranking stage
PREVIEW_CHARS = 160


@dataclass
class StageTrace:
    name: str
    took_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    query: str
    stages: list[StageTrace] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)  # the retrieval options used
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)  # chunk id -> what to show
    started: float = field(default_factory=time.perf_counter, repr=False)

    def add(self, name: str, started: float, **details: Any) -> StageTrace:
        stage = StageTrace(name, round((time.perf_counter() - started) * 1000, 2), details)
        self.stages.append(stage)
        return stage

    def remember(
        self,
        chunk_id: uuid.UUID,
        *,
        title: str,
        page: int | None,
        heading_path: list[str],
        text: str,
    ) -> None:
        """Keep enough about a passage that the trace can be read on its own."""
        self.chunks.setdefault(
            str(chunk_id),
            {
                "title": title,
                "page": page,
                "heading_path": list(heading_path),
                "preview": " ".join(text.split())[:PREVIEW_CHARS],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "query": self.query,
            "options": self.options,
            "total_ms": round((time.perf_counter() - self.started) * 1000, 2),
            "stages": [
                {"name": s.name, "took_ms": s.took_ms, "details": s.details} for s in self.stages
            ],
            "chunks": self.chunks,
        }


@dataclass(frozen=True)
class Passage:
    """Where a retrieved chunk sits in its document (for highlighting and citing)."""

    chunk_id: uuid.UUID
    char_start: int
    char_end: int


@dataclass
class ContextBlock:
    """One passage of the assembled context: a retrieved chunk, or the parent section or
    neighbouring chunks around it, with overlapping pieces merged into one exact text."""

    label: str  # "S1": how a prompt and an answer refer to it
    text: str
    document_id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    page: int | None
    heading_path: list[str]
    char_start: int
    char_end: int
    chunk_ids: list[uuid.UUID]  # every chunk whose text is inside the block
    retrieved_chunk_ids: list[uuid.UUID]  # the ones retrieval returned (what to highlight)
    score: float
    tokens: int
    notes: list[str] = field(default_factory=list)
    # the retrieved chunks inside the block with their own spans, so a citation of the block
    # can point at the exact passages that matched, not only at the whole block
    passages: list[Passage] = field(default_factory=list)


@dataclass
class ContextResult:
    blocks: list[ContextBlock]
    tokens: int
    budget: int


@dataclass
class RetrievalResult:
    hits: list[Hit]
    trace: RetrievalTrace
    warnings: list[str] = field(default_factory=list)  # degraded behaviour the caller should see
    context: ContextResult | None = None
    inferred: list = field(default_factory=list)  # filters read out of the question (self-query)
