"""Searching a workspace (Phase 3): ranked passages for a query, with the trace of how."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field, StringConstraints, model_validator
from sqlalchemy.exc import SQLAlchemyError

from app.core.auth import ViewerAccess
from app.core.deps import SessionDep, SettingsDep
from app.core.errors import BadRequest
from app.retrieve.catalogue import build_catalogue
from app.retrieve.filters import (
    MAX_CONDITIONS,
    FilterError,
    MetadataFilter,
    compile_filters,
)
from app.retrieve.history import save_retrieval
from app.retrieve.options import RetrievalOptions
from app.retrieve.pipeline import RetrievalPipeline
from app.retrieve.rerank import RerankUnavailable
from app.retrieve.scope import resolve_collections
from app.retrieve.transform import Turn
from app.retrieve.types import ContextResult, Hit, RetrievalTrace

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
    record: bool = Field(
        True, description="Save the search and its trace so the Inspector can show it later"
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
    ordinal: int
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
    schema_version: int
    query: str
    options: dict[str, Any]
    total_ms: float
    stages: list[StageOut]
    chunks: dict[str, dict[str, Any]] = Field(
        description="What to show for each passage the trace mentions, so it reads on its own"
    )


class PassageOut(BaseModel):
    chunk_id: uuid.UUID
    char_start: int
    char_end: int


class BlockOut(BaseModel):
    label: str
    text: str
    document_id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    page: int | None
    heading_path: list[str]
    char_start: int
    char_end: int
    chunk_ids: list[uuid.UUID]
    retrieved_chunk_ids: list[uuid.UUID]
    score: float
    tokens: int
    notes: list[str]
    passages: list[PassageOut]


class ContextOut(BaseModel):
    blocks: list[BlockOut]
    tokens: int
    budget: int


class SearchOut(BaseModel):
    query: str
    mode: str
    hits: list[HitOut]
    trace: TraceOut
    warnings: list[str]
    context: ContextOut | None = None
    inferred_filters: list[MetadataFilter] = []
    retrieval_id: uuid.UUID | None = Field(
        None, description="Where this search and its trace were saved (see /retrievals)"
    )


class FieldOut(BaseModel):
    name: str
    type: str
    ops: list[str]
    description: str
    values: list[Any] = Field(default_factory=list, description="Distinct values seen, if any")
    complete: bool = Field(False, description="True when `values` is every value there is")
    minimum: float | None = None
    maximum: float | None = None
    documents: int | None = Field(None, description="Documents that have this attribute")


def _hit_out(h: Hit) -> HitOut:
    return HitOut(**vars(h))


def block_out(b) -> BlockOut:
    return BlockOut(**{**vars(b), "passages": [vars(p) for p in b.passages]})


def _context_out(c: ContextResult | None) -> ContextOut | None:
    if c is None:
        return None
    return ContextOut(blocks=[block_out(b) for b in c.blocks], tokens=c.tokens, budget=c.budget)


def _trace_out(t: RetrievalTrace) -> TraceOut:
    return TraceOut(**t.to_dict())


@router.post("/workspaces/{workspace_id}/search", response_model=SearchOut)
async def search(
    body: SearchRequest,
    access: ViewerAccess,
    session: SessionDep,
    request: Request,
    settings: SettingsDep,
):
    """Find the passages that best match a query, across the workspace's collections (or only
    the ones named), narrowed by optional metadata filters. Each hit carries its scores and the
    reasons it was chosen; `trace` shows what every stage did."""
    collections = await resolve_collections(session, access.workspace.id, body.collection_ids)
    pipeline = RetrievalPipeline(session, request.app.state.providers, request.app.state.blobs)
    try:
        result = await pipeline.run(
            body.query,
            access.workspace.id,
            collections,
            RetrievalOptions(**body.model_dump(include=set(RetrievalOptions.model_fields))),
            body.filters,
            body.history,
        )
    except RerankUnavailable as e:
        raise BadRequest(str(e)) from None
    out = SearchOut(
        query=body.query,
        mode=body.mode,
        hits=[_hit_out(h) for h in result.hits],
        trace=_trace_out(result.trace),
        warnings=result.warnings,
        context=_context_out(result.context),
        inferred_filters=result.inferred,
    )
    if body.record:
        try:
            saved = await save_retrieval(
                session,
                workspace_id=access.workspace.id,
                user_id=access.user.id,
                response=out.model_dump(mode="json"),
                keep=settings.retrieval_history_limit,
            )
        except SQLAlchemyError:
            # A search that worked must not fail because its history could not be written.
            await session.rollback()
            out.warnings.append("The search could not be saved to the history")
        else:
            out.retrieval_id = saved.id if saved else None
    return out


@router.get("/workspaces/{workspace_id}/search/fields", response_model=list[FieldOut])
async def filterable_fields(access: ViewerAccess, session: SessionDep):
    """The fields a search can filter on: the built-in ones, and the document attributes that
    exist in this workspace (`attr.<name>`), each with the operators it accepts and the values
    seen. This is also what the self-query step shows the model."""
    catalogue = await build_catalogue(session, access.workspace.id)
    return [FieldOut(**vars(info)) for info in catalogue]
