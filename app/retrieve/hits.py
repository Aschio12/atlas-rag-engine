"""Building a `Hit` from a database row and the scores each stage gave it."""

from __future__ import annotations

from app.db.models import Chunk, Document
from app.ingest.sparse import terms
from app.ingest.vectors import lexical_text
from app.retrieve.candidates import Candidate
from app.retrieve.types import Hit

MAX_TERMS_SHOWN = 8


def matched_terms(query: str, chunk: Chunk) -> list[str]:
    """Query terms (as the sparse index sees them) that also occur in the passage."""
    shared = set(terms(query)) & set(terms(lexical_text(chunk)))
    return sorted(shared)[:MAX_TERMS_SHOWN]


def build_hit(
    chunk: Chunk,
    doc: Document,
    *,
    rank: int,
    score: float,
    best: Candidate,
    scores: dict[str, float],
    reasons: list[str],
) -> Hit:
    """`best` is the point that matched: when it is a question point, the passage was reached
    through a question generated for it."""
    payload = best.point.payload
    via_question = payload.get("kind") == "question"
    question = payload.get("question") if via_question else None
    if via_question:
        reasons = [*reasons, f"best match was a question this passage answers: {question!r}"]
    meta = chunk.meta or {}
    return Hit(
        chunk_id=chunk.id,
        document_id=doc.id,
        collection_id=doc.collection_id,
        rank=rank,
        ordinal=chunk.ordinal,
        score=score,
        text=chunk.text,
        title=doc.title,
        page=chunk.page,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        heading_path=list(meta.get("heading_path", [])),
        parent_id=chunk.parent_id,
        language=meta.get("language"),
        source_uri=doc.source_uri,
        matched="question" if via_question else "chunk",
        question=question,
        scores=scores,
        reasons=reasons,
    )
