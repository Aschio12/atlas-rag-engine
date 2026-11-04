"""Inspecting chunks (Phase 2.4): a document's stored chunks and a no-side-effects preview."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.api.documents import get_collection, get_document
from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import SessionDep
from app.core.errors import BadRequest
from app.db.models import Chunk, ChunkingStrategy
from app.ingest.chunking import ChunkDraft, ChunkingConfig, build_chunker
from app.ingest.parsers import ParseError, parse_markdown
from app.ingest.parsers.plain import parse_text

router = APIRouter(tags=["chunks"])

MAX_PREVIEW_CHARS = 200_000


class ChunkOut(BaseModel):
    id: uuid.UUID | None
    ordinal: int
    kind: str
    text: str
    char_start: int
    char_end: int
    page: int | None
    token_count: int
    parent_id: uuid.UUID | None
    heading_path: list[str]
    context: str | None
    language: str | None = None
    keywords: list[str] = []
    summary: str | None = None
    questions: list[str] = []


class ChunkList(BaseModel):
    items: list[ChunkOut]
    total: int


class PreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_PREVIEW_CHARS)
    format: Literal["text", "markdown"] = "markdown"
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    config: ChunkingConfig = Field(default_factory=ChunkingConfig)


class PreviewStats(BaseModel):
    strategy: ChunkingStrategy
    chunks: int
    parents: int
    avg_tokens: int
    max_tokens: int
    characters: int


class PreviewOut(BaseModel):
    stats: PreviewStats
    items: list[ChunkOut]


def _from_row(c: Chunk) -> ChunkOut:
    meta = c.meta or {}
    return ChunkOut(
        id=c.id,
        ordinal=c.ordinal,
        kind=meta.get("kind", "chunk"),
        text=c.text,
        char_start=c.char_start,
        char_end=c.char_end,
        page=c.page,
        token_count=c.token_count,
        parent_id=c.parent_id,
        heading_path=meta.get("heading_path", []),
        context=meta.get("context"),
        language=meta.get("language"),
        keywords=meta.get("keywords", []),
        summary=meta.get("summary"),
        questions=meta.get("questions", []),
    )


@router.get(
    "/workspaces/{workspace_id}/collections/{collection_id}/documents/{document_id}/chunks",
    response_model=ChunkList,
)
async def list_chunks(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    access: ViewerAccess,
    session: SessionDep,
    kind: Literal["chunk", "parent", "child"] | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """The chunks stored for a document, in order. Parents (parent-child strategy) are listed
    too; filter with `kind`."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    document = await get_document(session, collection, document_id)
    base = select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.ordinal)
    if kind is None:  # page in the database
        total = (
            await session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
            )
        ).scalar_one()
        rows = (await session.execute(base.limit(limit).offset(offset))).scalars().all()
        return ChunkList(items=[_from_row(c) for c in rows], total=total)
    # `kind` lives in JSON metadata, so filter in Python: a document has at most a few thousand.
    every = (await session.execute(base)).scalars().all()
    wanted = [c for c in every if (c.meta or {}).get("kind", "chunk") == kind]
    return ChunkList(
        items=[_from_row(c) for c in wanted[offset : offset + limit]], total=len(wanted)
    )


def _draft_out(index: int, d: ChunkDraft) -> ChunkOut:
    return ChunkOut(
        id=None,
        ordinal=index,
        kind=d.kind,
        text=d.text,
        char_start=d.char_start,
        char_end=d.char_end,
        page=d.page,
        token_count=d.token_count,
        parent_id=None,
        heading_path=d.heading_path,
        context=d.context,
    )


@router.post("/workspaces/{workspace_id}/chunking/preview", response_model=PreviewOut)
async def preview_chunking(body: PreviewRequest, access: MemberAccess, request: Request):
    """Chunk some text with a strategy and settings and show the result, storing nothing. Use it
    to pick sizes before creating a collection. `contextual` is not offered here because it
    would spend LLM calls on every request."""
    if body.strategy is ChunkingStrategy.CONTEXTUAL:
        raise BadRequest("The contextual strategy is not available in preview: it calls the LLM")
    data = body.text.encode("utf-8")
    try:
        parsed = parse_markdown(data) if body.format == "markdown" else parse_text(data)
    except ParseError as e:
        raise BadRequest(str(e)) from None
    chunker = build_chunker(body.strategy, request.app.state.providers)
    drafts = await chunker.chunk(parsed, body.config)
    searchable = [d for d in drafts if d.indexable]
    stats = PreviewStats(
        strategy=body.strategy,
        chunks=len(searchable),
        parents=len(drafts) - len(searchable),
        avg_tokens=round(sum(d.token_count for d in searchable) / max(len(searchable), 1)),
        max_tokens=max((d.token_count for d in searchable), default=0),
        characters=len(parsed.text),
    )
    return PreviewOut(stats=stats, items=[_draft_out(i, d) for i, d in enumerate(drafts)])
