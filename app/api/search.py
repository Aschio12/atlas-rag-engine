"""Searching a workspace (Phase 3): ranked passages for a query, with the trace of how."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, StringConstraints, model_validator
from sqlalchemy import select

from app.core.auth import ViewerAccess
from app.core.deps import SessionDep
from app.core.errors import BadRequest, NotFound
from app.db.models import Collection
from app.retrieve.filters import (
    FIELDS,
    MAX_CONDITIONS,
    CompiledFilter,
    FilterError,
    MetadataFilter,
    compile_filters,
)
from app.retrieve.options import RetrievalOptions
from app.retrieve.pipeline import RetrievalPipeline
from app.retrieve.rerank import RerankUnavailable
from app.retrieve.transform import Turn
from app.retrieve.types import Hit, RetrievalTrace

router = APIRouter(tags=["search"])

MAX_QUERY_CHARS = 2000
MAX_COLLECTIONS = 50

Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
]


class SearchRequest(RetrievalOptions):
    query: Query
    collection_ids: list[uuid.UUID] | None = Field(
        None, max_length=MAX_COLLECTIONS, description="Search only these; default: all of them"
    )
    history: list[Turn] = Field(
        default_factory=list,
        max_length=20,
        description="Earlier turns, oldest first; used by the 'rewrite' transform",
    )
    filters: list[MetadataFilter] = Field(
        default_factory=list, max_length=MAX_CONDITIONS, description="All must hold"
    )

    @model_validator(mode="after")
    def _filters_are_consistent(self) -> SearchRequest:
        try:
            compile_filters(self.filters)
        except FilterError as e:
            raise ValueError(str(e)) from None
        return self


class HitOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    collection_id: uuid.UUID
    rank: int
    score: float
    text: str
    title: str
    page: int | None
    char_start: int
    char_end: int
    heading_path: list[str]
    parent_id: uuid.UUID | None
    language: str | None
    source_uri: str | None
    matched: Literal["chunk", "question"]
    question: str | None
    scores: dict[str, float]
    reasons: list[str]


class StageOut(BaseModel):
    name: str
    took_ms: float
    details: dict[str, Any]


class TraceOut(BaseModel):
    query: str
    stages: list[StageOut]


class SearchOut(BaseModel):
    query: str
    mode: str
    hits: list[HitOut]
    trace: TraceOut
    warnings: list[str]


class FieldOut(BaseModel):
    name: str
    type: str
    ops: list[str]
    description: str


def _hit_out(h: Hit) -> HitOut:
    return HitOut(**vars(h))


def _trace_out(t: RetrievalTrace) -> TraceOut:
    return TraceOut(
        query=t.query,
        stages=[StageOut(name=s.name, took_ms=s.took_ms, details=s.details) for s in t.stages],
    )


async def _collections(
    session, workspace_id: uuid.UUID, wanted: list[uuid.UUID] | None
) -> list[Collection]:
    query = (
        select(Collection).where(Collection.workspace_id == workspace_id).order_by(Collection.name)
    )
    rows = list((await session.execute(query)).scalars())
    if wanted is None:
        return rows
    by_id = {c.id: c for c in rows}
    chosen = list(dict.fromkeys(wanted))
    if missing := [i for i in chosen if i not in by_id]:  # also covers other workspaces' ids
        raise NotFound(f"Collection not found: {missing[0]}")
    return [by_id[i] for i in chosen]


@router.post("/workspaces/{workspace_id}/search", response_model=SearchOut)
async def search(body: SearchRequest, access: ViewerAccess, session: SessionDep, request: Request):
    """Find the passages that best match a query, across the workspace's collections (or only
    the ones named), narrowed by optional metadata filters. Each hit carries its scores and the
    reasons it was chosen; `trace` shows what every stage did."""
    collections = await _collections(session, access.workspace.id, body.collection_ids)
    pipeline = RetrievalPipeline(session, request.app.state.providers)
    try:
        result = await pipeline.run(
            body.query,
            access.workspace.id,
            collections,
            RetrievalOptions(**body.model_dump(include=set(RetrievalOptions.model_fields))),
            compile_filters(body.filters) if body.filters else CompiledFilter(),
            body.history,
        )
    except RerankUnavailable as e:
        raise BadRequest(str(e)) from None
    return SearchOut(
        query=body.query,
        mode=body.mode,
        hits=[_hit_out(h) for h in result.hits],
        trace=_trace_out(result.trace),
        warnings=result.warnings,
    )


@router.get("/workspaces/{workspace_id}/search/fields", response_model=list[FieldOut])
async def filterable_fields(access: ViewerAccess):
    """The fields a search can filter on, and the operators each accepts."""
    return [
        FieldOut(name=name, type=spec.kind, ops=list(spec.ops), description=spec.description)
        for name, spec in FIELDS.items()
    ]
