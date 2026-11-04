"""What retrieval returns: ranked hits and a trace of how they were found.

A `Hit` is one passage with the score each stage gave it and the reasons it is here; the trace
records what every stage did (Phase 3.8 extends it to the whole pipeline for the Inspector).
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


@dataclass
class StageTrace:
    name: str
    took_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTrace:
    query: str
    stages: list[StageTrace] = field(default_factory=list)

    def add(self, name: str, started: float, **details: Any) -> StageTrace:
        stage = StageTrace(name, round((time.perf_counter() - started) * 1000, 2), details)
        self.stages.append(stage)
        return stage


@dataclass
class RetrievalResult:
    hits: list[Hit]
    trace: RetrievalTrace
