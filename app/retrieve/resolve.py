"""From points to passages: checking candidates against the database (Phase 3.1).

Passage text is read from the database, never from the vector store, and a point is served only
while its chunk exists in the collection the point claims and its `index_version` is the one the
document's record says is current. Points left behind by an interrupted or superseded indexing
run are skipped. That check comes first, so such a point can never hide a valid one.

A chunk can then be reached through several points: its own, and one per question generated for
it (Phase 2.5). Those collapse to the chunk, keeping the best score.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.base import ScoredPoint
from app.retrieve.candidates import Candidate

Row = tuple[Chunk, Document]


def chunk_id_of(point: ScoredPoint) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(point.payload.get("chunk_id") or point.id))
    except ValueError:
        return None


async def load_rows(session: AsyncSession, chunk_ids: set[uuid.UUID]) -> dict[uuid.UUID, Row]:
    rows: dict[uuid.UUID, Row] = {}
    ids = sorted(chunk_ids)
    for start in range(0, len(ids), 500):  # keep the IN list well under any driver limit
        result = await session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(ids[start : start + 500]))
        )
        rows.update({chunk.id: (chunk, doc) for chunk, doc in result.all()})
    return rows


def is_current(cand: Candidate, row: Row | None) -> str | None:
    """None if the point may be served, otherwise why not: "missing" (no such chunk, or it
    belongs to another collection than the point claims) or "stale" (an older indexing run)."""
    if row is None or row[1].collection_id != cand.collection.id:
        return "missing"
    version = ((row[1].meta or {}).get("indexing") or {}).get("version")
    if version is None or cand.point.payload.get("index_version") != version:
        return "stale"
    return None


def collapse(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Sort by score and keep the best point per chunk. Returns the survivors and how many
    points were folded into another point of the same chunk."""
    best: dict[uuid.UUID, Candidate] = {}
    folded = 0
    for cand in sorted(candidates, key=lambda c: (-c.point.score, c.point.id)):
        chunk_id = chunk_id_of(cand.point)
        if chunk_id is None:
            continue
        if chunk_id in best:
            folded += 1
        else:
            best[chunk_id] = cand
    return list(best.values()), folded


def resolve(
    candidates: list[Candidate], rows: dict[uuid.UUID, Row]
) -> tuple[list[Candidate], dict[str, int]]:
    """Keep the servable candidates, one per chunk, best first, with counts of what was
    dropped (`missing`, `stale`) or folded into another point of the same chunk."""
    stats = {"missing": 0, "stale": 0}
    valid: list[Candidate] = []
    for cand in candidates:
        reason = is_current(cand, rows.get(chunk_id_of(cand.point)))
        if reason:
            stats[reason] += 1
        else:
            valid.append(cand)
    kept, folded = collapse(valid)
    return kept, {**stats, "folded_into_chunk": folded}
