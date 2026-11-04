"""Dense (embedding) retrieval: top-k nearest passages for a query (Phase 3.1).

The query is embedded once per embedding model in play (each collection fixes its own model),
every collection is searched in its own vector collection with the workspace and collection
filters always applied on top of the caller's metadata filters, and the candidates are merged
by similarity.

Passage text is read from the database, never from the vector store, and a point is served only
while its `index_version` is the one the document's record says is current; points left behind
by an interrupted or superseded indexing run are skipped. That check comes first, so such a point
can never hide a valid one.

A chunk can then be reached through several points: its own, and one per question generated for
it (Phase 2.5). Those collapse to the chunk, keeping the best score and noting which point won.

Scores of different embedding models are not comparable. A search across collections that use
different models still merges by raw similarity, and says so in the trace.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Collection, Document
from app.ingest.vectors import vector_collection
from app.providers.base import ScoredPoint
from app.providers.registry import Providers
from app.retrieve.filters import CompiledFilter
from app.retrieve.types import Hit, RetrievalResult, RetrievalTrace

OVERFETCH = 3  # candidates per requested hit: duplicates, stale points and misses are dropped
MIN_FETCH = 10
MAX_FETCH = 200


@dataclass(frozen=True)
class Candidate:
    point: ScoredPoint
    collection: Collection


def fetch_size(top_k: int) -> int:
    return min(max(top_k * OVERFETCH, MIN_FETCH), MAX_FETCH)


def _chunk_id(point: ScoredPoint) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(point.payload.get("chunk_id") or point.id))
    except ValueError:
        return None


def collapse(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Sort by score and keep the best point per chunk. Returns the survivors and how many
    points were folded into another point of the same chunk."""
    best: dict[uuid.UUID, Candidate] = {}
    folded = 0
    for cand in sorted(candidates, key=lambda c: (-c.point.score, c.point.id)):
        chunk_id = _chunk_id(cand.point)
        if chunk_id is None:
            continue
        if chunk_id in best:
            folded += 1
        else:
            best[chunk_id] = cand
    return list(best.values()), folded


async def load_rows(
    session: AsyncSession, chunk_ids: set[uuid.UUID]
) -> dict[uuid.UUID, tuple[Chunk, Document]]:
    rows: dict[uuid.UUID, tuple[Chunk, Document]] = {}
    ids = sorted(chunk_ids)
    for start in range(0, len(ids), 500):  # keep the IN list well under any driver limit
        result = await session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(ids[start : start + 500]))
        )
        rows.update({chunk.id: (chunk, doc) for chunk, doc in result.all()})
    return rows


def is_current(cand: Candidate, row: tuple[Chunk, Document] | None) -> str | None:
    """None if the point may be served, otherwise why not: "missing" (no such chunk, or it
    belongs to another collection than the point claims) or "stale" (an older indexing run)."""
    if row is None or row[1].collection_id != cand.collection.id:
        return "missing"
    version = ((row[1].meta or {}).get("indexing") or {}).get("version")
    if version is None or cand.point.payload.get("index_version") != version:
        return "stale"
    return None


class DenseRetriever:
    def __init__(self, session: AsyncSession, providers: Providers):
        self._session = session
        self._providers = providers

    async def search(
        self,
        query: str,
        workspace_id: uuid.UUID,
        collections: list[Collection],
        *,
        top_k: int = 10,
        min_score: float | None = None,
        metadata: CompiledFilter | None = None,
    ) -> RetrievalResult:
        if any(c.workspace_id != workspace_id for c in collections):
            raise ValueError("every searched collection must belong to the workspace")
        metadata = metadata or CompiledFilter()
        trace = RetrievalTrace(query)
        started = time.perf_counter()
        models = sorted({c.embed_model for c in collections})
        trace.add(
            "scope",
            started,
            workspace_id=str(workspace_id),
            collections=[
                {"id": str(c.id), "name": c.name, "model": c.embed_model} for c in collections
            ],
            filters=list(metadata.summary),
        )
        if not collections:
            return RetrievalResult([], trace)

        started = time.perf_counter()
        vectors: dict[str, list[float]] = {}
        for model in models:
            (vectors[model],) = await self._providers.embedder_for(model).embed([query])
        trace.add("embed_query", started, models=models)

        started = time.perf_counter()
        limit = fetch_size(top_k)
        store = self._providers.vector_store

        async def search_one(col: Collection) -> list[Candidate]:
            # The scope is applied last so nothing in the caller's filters can widen it.
            filters = {
                **metadata.filters,
                "workspace_id": str(workspace_id),
                "collection_id": str(col.id),
            }
            found = await store.search(
                vector_collection(col.id),
                vectors[col.embed_model],
                limit,
                filters=filters,
                exclude=metadata.exclude or None,
            )
            return [Candidate(p, col) for p in found]

        per_collection = await asyncio.gather(*(search_one(c) for c in collections))
        candidates = [c for found in per_collection for c in found]
        trace.add(
            "dense_search",
            started,
            fetched_per_collection=limit,
            points_found={
                str(c.id): len(f) for c, f in zip(collections, per_collection, strict=True)
            },
            comparable_scores=len(models) == 1,
        )

        started = time.perf_counter()
        rows = await load_rows(self._session, {i for c in candidates if (i := _chunk_id(c.point))})
        dropped = {"missing": 0, "stale": 0}
        valid: list[Candidate] = []
        for cand in candidates:
            reason = is_current(cand, rows.get(_chunk_id(cand.point)))
            if reason:
                dropped[reason] += 1
            else:
                valid.append(cand)
        survivors, folded = collapse(valid)
        if min_score is not None:
            kept = [c for c in survivors if c.point.score >= min_score]
            below, survivors = len(survivors) - len(kept), kept
        else:
            below = 0
        hits = [
            _to_hit(*rows[_chunk_id(c.point)], cand=c, rank=i)
            for i, c in enumerate(survivors[:top_k], start=1)
        ]
        trace.add(
            "resolve",
            started,
            folded_into_chunk=folded,
            below_min_score=below,
            returned=len(hits),
            **dropped,
        )
        return RetrievalResult(hits, trace)


def _to_hit(chunk: Chunk, doc: Document, cand: Candidate, rank: int) -> Hit:
    payload = cand.point.payload
    via_question = payload.get("kind") == "question"
    question = payload.get("question") if via_question else None
    reasons = [f"dense cosine similarity {cand.point.score:.4f}"]
    if via_question:
        reasons.append(f"best match was a question this passage answers: {question!r}")
    return Hit(
        chunk_id=chunk.id,
        document_id=doc.id,
        collection_id=doc.collection_id,
        rank=rank,
        score=cand.point.score,
        text=chunk.text,
        title=doc.title,
        page=chunk.page,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        heading_path=list((chunk.meta or {}).get("heading_path", [])),
        parent_id=chunk.parent_id,
        language=(chunk.meta or {}).get("language"),
        source_uri=doc.source_uri,
        matched="question" if via_question else "chunk",
        question=question,
        scores={"dense": cand.point.score},
        reasons=reasons,
    )
