"""Sources: connect a website or a GitHub repository to a collection and keep it in sync
(Phase 2.7)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.api.documents import get_collection
from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import BlobsDep, SessionDep, SettingsDep
from app.core.errors import BadRequest, Conflict, NotFound
from app.db.models import Collection, Document, Source, SourceStatus, SourceType
from app.ingest.connectors.github import GithubSourceConfig
from app.ingest.connectors.web import WebSourceConfig
from app.ingest.fetch import FetchError, validate_url
from app.ingest.purge import delete_documents

router = APIRouter(
    prefix="/workspaces/{workspace_id}/collections/{collection_id}/sources", tags=["sources"]
)


class WebSourceCreate(BaseModel):
    type: Literal["web"]
    name: str | None = Field(default=None, max_length=200)
    config: WebSourceConfig


class GithubSourceCreate(BaseModel):
    type: Literal["github"]
    name: str | None = Field(default=None, max_length=200)
    config: GithubSourceConfig


SourceCreate = Annotated[WebSourceCreate | GithubSourceCreate, Field(discriminator="type")]


class SourceOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    type: SourceType
    name: str
    config: dict[str, Any]
    status: SourceStatus
    stats: dict[str, Any]
    error: str | None
    document_count: int
    started_at: datetime | None
    finished_at: datetime | None
    last_synced_at: datetime | None
    created_at: datetime


class SourceList(BaseModel):
    items: list[SourceOut]
    total: int


async def _counts(session, source_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    if not source_ids:
        return {}
    rows = await session.execute(
        select(Document.source_id, func.count())
        .where(Document.source_id.in_(source_ids))
        .group_by(Document.source_id)
    )
    return {sid: n for sid, n in rows.all()}


def _out(source: Source, count: int) -> SourceOut:
    return SourceOut(
        id=source.id,
        collection_id=source.collection_id,
        type=source.type,
        name=source.name,
        config=source.config,
        status=source.status,
        stats=source.stats or {},
        error=source.error,
        document_count=count,
        started_at=source.started_at,
        finished_at=source.finished_at,
        last_synced_at=source.last_synced_at,
        created_at=source.created_at,
    )


async def _get(session, collection: Collection, source_id: uuid.UUID) -> Source:
    source = (
        await session.execute(
            select(Source).where(Source.id == source_id, Source.collection_id == collection.id)
        )
    ).scalar_one_or_none()
    if source is None:
        raise NotFound("Source not found")
    return source


def _default_name(body: WebSourceCreate | GithubSourceCreate) -> str:
    if isinstance(body, GithubSourceCreate):
        return body.config.repo
    parts = urlsplit(body.config.url)
    return (parts.hostname or "site") + (parts.path.rstrip("/") or "")


@router.post("", status_code=202, response_model=SourceOut)
async def create_source(
    request: Request,
    collection_id: uuid.UUID,
    body: SourceCreate,
    access: MemberAccess,
    session: SessionDep,
    settings: SettingsDep,
):
    """Connect a source and start its first sync (returns immediately; poll the source)."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    existing = (
        await session.execute(
            select(func.count()).select_from(Source).where(Source.collection_id == collection.id)
        )
    ).scalar_one()
    if existing >= settings.source_max_per_collection:
        raise Conflict(
            f"A collection can have at most {settings.source_max_per_collection} sources"
        )
    if isinstance(body, WebSourceCreate):
        try:  # refuse internal addresses now, not minutes later in the background
            await validate_url(
                body.config.url,
                allow_private=settings.allow_private_urls,
                resolver=request.app.state.url_resolver,
            )
        except FetchError as e:
            raise BadRequest(str(e)) from None
    name = (body.name or "").strip() or _default_name(body)
    source = Source(
        collection_id=collection.id,
        type=SourceType(body.type),
        name=name[:200],
        config=body.config.model_dump(),
        status=SourceStatus.IDLE,
        stats={},
    )
    session.add(source)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise Conflict(f"This collection already has a source named '{name}'") from None
    await session.refresh(source)
    request.app.state.sync_runner.start(source.id)
    return _out(source, 0)


@router.get("", response_model=SourceList)
async def list_sources(collection_id: uuid.UUID, access: ViewerAccess, session: SessionDep):
    collection = await get_collection(session, access.workspace.id, collection_id)
    rows = (
        (
            await session.execute(
                select(Source)
                .where(Source.collection_id == collection.id)
                .order_by(Source.created_at.desc(), Source.id)
            )
        )
        .scalars()
        .all()
    )
    counts = await _counts(session, [s.id for s in rows])
    return SourceList(items=[_out(s, counts.get(s.id, 0)) for s in rows], total=len(rows))


@router.get("/{source_id}", response_model=SourceOut)
async def get_source(
    collection_id: uuid.UUID, source_id: uuid.UUID, access: ViewerAccess, session: SessionDep
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    source = await _get(session, collection, source_id)
    return _out(source, (await _counts(session, [source.id])).get(source.id, 0))


@router.post("/{source_id}/sync", status_code=202, response_model=SourceOut)
async def sync_source(
    request: Request,
    collection_id: uuid.UUID,
    source_id: uuid.UUID,
    access: MemberAccess,
    session: SessionDep,
):
    """Run the sync again: new items are added, changed ones updated, vanished ones removed."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    source = await _get(session, collection, source_id)
    runner = request.app.state.sync_runner
    if runner.is_running(source.id):
        raise Conflict("A sync is already running for this source")
    runner.start(source.id)
    count = (await _counts(session, [source.id])).get(source.id, 0)
    return _out(source, count)


@router.delete("/{source_id}", status_code=204)
async def delete_source(
    request: Request,
    collection_id: uuid.UUID,
    source_id: uuid.UUID,
    access: MemberAccess,
    session: SessionDep,
    blobs: BlobsDep,
    delete_documents_too: Annotated[
        bool, Query(alias="delete_documents", description="Also delete the documents it created")
    ] = False,
):
    """Disconnect a source. Its documents are kept unless `delete_documents=true`."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    source = await _get(session, collection, source_id)
    await request.app.state.sync_runner.cancel(source.id)
    if delete_documents_too:
        ids = (
            (await session.execute(select(Document.id).where(Document.source_id == source.id)))
            .scalars()
            .all()
        )
        await delete_documents(session, request.app.state.providers, blobs, collection, list(ids))
        source = await _get(session, collection, source_id)
    await session.delete(source)
    await session.commit()
    return Response(status_code=204)
