"""Collections: named sets of documents inside a workspace (rag1's "topic cards")."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AdminAccess, MemberAccess, ViewerAccess
from app.core.deps import BlobsDep, SessionDep, SettingsDep
from app.core.errors import Conflict, NotFound
from app.db.models import ChunkingStrategy, Collection, Document
from app.ingest import keys
from app.ingest.chunking import ChunkingConfig
from app.ingest.enrichment.config import EnrichmentConfig
from app.ingest.purge import purge_collection

router = APIRouter(prefix="/workspaces/{workspace_id}/collections", tags=["collections"])


def _clean(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120, examples=["Product manuals"])
    description: str = Field(default="", max_length=2000)
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunking_config: ChunkingConfig = Field(
        default_factory=ChunkingConfig,
        description="Chunk sizes and strategy options. Fixed after creation.",
    )
    enrichment: EnrichmentConfig = Field(
        default_factory=EnrichmentConfig,
        description="Per-chunk enrichment (language, keywords, optional LLM summaries and "
        "hypothetical questions).",
    )
    embed_model: str | None = Field(
        default=None,
        max_length=200,
        description="Defaults to the server's embedding model. Fixed after creation.",
    )

    _name = field_validator("name")(_clean)


class CollectionUpdate(BaseModel):
    """Only name and description can change: chunking strategy and embedding model determine
    what is stored in the index, so changing them requires re-indexing (Phase 2.8)."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return None if v is None else _clean(v)

    @model_validator(mode="after")
    def _something_to_change(self) -> CollectionUpdate:
        if self.name is None and self.description is None:
            raise ValueError("provide at least one of: name, description")
        return self


class CollectionOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str
    description: str
    chunking_strategy: ChunkingStrategy
    chunking_config: ChunkingConfig
    enrichment: EnrichmentConfig
    embed_model: str
    document_count: int
    created_at: datetime
    updated_at: datetime


class CollectionList(BaseModel):
    items: list[CollectionOut]
    total: int


def to_out(c: Collection, document_count: int) -> CollectionOut:
    return CollectionOut(
        id=c.id,
        workspace_id=c.workspace_id,
        name=c.name,
        description=c.description,
        chunking_strategy=c.chunking_strategy,
        chunking_config=ChunkingConfig(**(c.chunking_config or {})),
        enrichment=EnrichmentConfig(**(c.enrichment_config or {})),
        embed_model=c.embed_model,
        document_count=document_count,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _with_counts(workspace_id: uuid.UUID):
    return (
        select(Collection, func.count(Document.id))
        .outerjoin(Document, Document.collection_id == Collection.id)
        .where(Collection.workspace_id == workspace_id)
        .group_by(Collection.id)
    )


async def _get(session: AsyncSession, workspace_id: uuid.UUID, collection_id: uuid.UUID):
    # Always filtered by workspace_id: a collection id from another tenant is simply "not found".
    row = (
        await session.execute(_with_counts(workspace_id).where(Collection.id == collection_id))
    ).one_or_none()
    if row is None:
        raise NotFound("Collection not found")
    return row


async def _name_taken(
    session: AsyncSession, workspace_id: uuid.UUID, name: str, *, exclude: uuid.UUID | None = None
) -> bool:
    q = select(Collection.id).where(
        Collection.workspace_id == workspace_id, func.lower(Collection.name) == name.lower()
    )
    if exclude is not None:
        q = q.where(Collection.id != exclude)
    return (await session.execute(q)).first() is not None


def _duplicate(name: str) -> Conflict:
    return Conflict(f"A collection named '{name}' already exists in this workspace")


@router.post("", status_code=201, response_model=CollectionOut)
async def create_collection(
    body: CollectionCreate, access: MemberAccess, session: SessionDep, settings: SettingsDep
):
    if await _name_taken(session, access.workspace.id, body.name):
        raise _duplicate(body.name)
    collection = Collection(
        workspace_id=access.workspace.id,
        name=body.name,
        description=body.description,
        chunking_strategy=body.chunking_strategy,
        chunking_config=body.chunking_config.model_dump(),
        enrichment_config=body.enrichment.model_dump(),
        embed_model=body.embed_model or settings.resolved_embed_model,
    )
    session.add(collection)
    try:
        await session.commit()
    except IntegrityError:
        raise _duplicate(body.name) from None
    await session.refresh(collection)
    return to_out(collection, 0)


@router.get("", response_model=CollectionList)
async def list_collections(
    access: ViewerAccess,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    total = (
        await session.execute(
            select(func.count())
            .select_from(Collection)
            .where(Collection.workspace_id == access.workspace.id)
        )
    ).scalar_one()
    rows = (
        await session.execute(
            _with_counts(access.workspace.id)
            .order_by(Collection.created_at.desc(), Collection.id)
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return CollectionList(items=[to_out(c, n) for c, n in rows], total=total)


@router.get("/{collection_id}", response_model=CollectionOut)
async def get_collection(collection_id: uuid.UUID, access: ViewerAccess, session: SessionDep):
    collection, count = await _get(session, access.workspace.id, collection_id)
    return to_out(collection, count)


@router.patch("/{collection_id}", response_model=CollectionOut)
async def update_collection(
    collection_id: uuid.UUID, body: CollectionUpdate, access: MemberAccess, session: SessionDep
):
    collection, count = await _get(session, access.workspace.id, collection_id)
    if body.name is not None and body.name.lower() != collection.name.lower():
        if await _name_taken(session, access.workspace.id, body.name, exclude=collection.id):
            raise _duplicate(body.name)
    if body.name is not None:
        collection.name = body.name
    if body.description is not None:
        collection.description = body.description
    try:
        await session.commit()
    except IntegrityError:
        raise _duplicate(body.name or collection.name) from None
    await session.refresh(collection)
    return to_out(collection, count)


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    request: Request,
    collection_id: uuid.UUID,
    access: AdminAccess,
    session: SessionDep,
    blobs: BlobsDep,
) -> Response:
    """Delete a collection with its documents, chunks, vectors and stored files (admin/owner)."""
    collection, _ = await _get(session, access.workspace.id, collection_id)
    prefix = keys.collection_prefix(collection.workspace_id, collection.id)
    await purge_collection(request.app.state.providers, collection.id)  # 502 if the index is down
    await session.delete(collection)
    await session.commit()
    await blobs.delete_prefix(prefix)  # after the commit: orphaned files are harmless
    return Response(status_code=204)
