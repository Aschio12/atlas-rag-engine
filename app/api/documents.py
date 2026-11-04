"""Documents: upload, list, inspect, download and delete (Phase 2.1)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, File, Form, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import BlobsDep, SessionDep, SettingsDep
from app.core.errors import (
    BadRequest,
    NotFound,
    PayloadTooLarge,
    UpstreamError,
)
from app.db.models import Collection, Document, DocumentStatus, IngestJob, JobState
from app.ingest import keys
from app.ingest.fetch import FetchError
from app.ingest.parsers import ParsedDocument
from app.ingest.purge import purge_document
from app.ingest.service import create_document
from app.storage import BlobNotFound

log = logging.getLogger("atlas.documents")

router = APIRouter(
    prefix="/workspaces/{workspace_id}/collections/{collection_id}/documents",
    tags=["documents"],
)

MAX_TEXT_CHARS = 5_000_000
READ_CHUNK = 1024 * 1024


class JobBrief(BaseModel):
    id: uuid.UUID
    state: JobState
    progress: int
    attempts: int
    error: str | None


class DocumentOut(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    source_type: str
    source_uri: str | None
    filename: str | None
    kind: str | None
    size_bytes: int
    content_hash: str
    status: DocumentStatus
    metadata: dict[str, Any]
    job: JobBrief | None = None
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class TextDocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500, examples=["Refund policy"])
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    format: Literal["text", "markdown"] = "text"

    @field_validator("title")
    @classmethod
    def _title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class UrlDocumentCreate(BaseModel):
    url: str = Field(min_length=1, max_length=2000, examples=["https://example.com/docs/faq.html"])
    title: str | None = Field(default=None, max_length=500)


def to_out(d: Document, job: IngestJob | None = None) -> DocumentOut:
    meta = d.meta or {}
    return DocumentOut(
        id=d.id,
        collection_id=d.collection_id,
        title=d.title,
        source_type=d.source_type,
        source_uri=d.source_uri,
        filename=meta.get("filename"),
        kind=meta.get("kind"),
        size_bytes=d.size_bytes,
        content_hash=d.content_hash,
        status=d.status,
        metadata=meta,
        job=JobBrief.model_validate(job, from_attributes=True) if job else None,
        created_at=d.created_at,
        updated_at=d.updated_at,
    )


async def get_collection(
    session: AsyncSession, workspace_id: uuid.UUID, collection_id: uuid.UUID
) -> Collection:
    # Filtered by workspace: a collection id from another tenant is simply "not found".
    c = (
        await session.execute(
            select(Collection).where(
                Collection.id == collection_id, Collection.workspace_id == workspace_id
            )
        )
    ).scalar_one_or_none()
    if c is None:
        raise NotFound("Collection not found")
    return c


async def get_document(
    session: AsyncSession, collection: Collection, document_id: uuid.UUID
) -> Document:
    d = (
        await session.execute(
            select(Document).where(
                Document.id == document_id, Document.collection_id == collection.id
            )
        )
    ).scalar_one_or_none()
    if d is None:
        raise NotFound("Document not found")
    return d


async def latest_jobs(
    session: AsyncSession, document_ids: list[uuid.UUID]
) -> dict[uuid.UUID, IngestJob]:
    """The most recent job of each document, in one query."""
    if not document_ids:
        return {}
    rows = (
        (
            await session.execute(
                select(IngestJob)
                .where(IngestJob.document_id.in_(document_ids))
                .order_by(IngestJob.created_at, IngestJob.id)
            )
        )
        .scalars()
        .all()
    )
    return {j.document_id: j for j in rows}  # later rows overwrite earlier ones


async def read_limited(file: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing to buffer more than `limit` bytes."""
    buf = bytearray()
    while chunk := await file.read(READ_CHUNK):
        buf += chunk
        if len(buf) > limit:
            raise PayloadTooLarge(f"The file is larger than the {limit // (1024 * 1024)} MB limit")
    return bytes(buf)


@router.post("", status_code=201, response_model=DocumentOut)
async def upload_document(
    request: Request,
    collection_id: uuid.UUID,
    access: MemberAccess,
    session: SessionDep,
    settings: SettingsDep,
    blobs: BlobsDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX, PPTX, HTML, Markdown, CSV or text")],
    title: Annotated[str | None, Form(max_length=500)] = None,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    data = await read_limited(file, settings.max_upload_bytes)
    doc, job = await create_document(
        session,
        blobs,
        collection,
        data=data,
        filename=file.filename,
        content_type=file.content_type,
        source_type="file",
        title=title,
    )
    request.app.state.worker.wake()
    return to_out(doc, job)


@router.post("/text", status_code=201, response_model=DocumentOut)
async def create_text_document(
    request: Request,
    collection_id: uuid.UUID,
    body: TextDocumentCreate,
    access: MemberAccess,
    session: SessionDep,
    settings: SettingsDep,
    blobs: BlobsDep,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    data = body.text.encode("utf-8")
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLarge(f"The text is larger than the {settings.max_upload_mb} MB limit")
    ext = "md" if body.format == "markdown" else "txt"
    doc, job = await create_document(
        session,
        blobs,
        collection,
        data=data,
        filename=f"{body.title}.{ext}",
        content_type=None,
        source_type="text",
        title=body.title,
    )
    request.app.state.worker.wake()
    return to_out(doc, job)


@router.post("/url", status_code=201, response_model=DocumentOut)
async def create_url_document(
    request: Request,
    collection_id: uuid.UUID,
    body: UrlDocumentCreate,
    access: MemberAccess,
    session: SessionDep,
    settings: SettingsDep,
    blobs: BlobsDep,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    try:
        fetched = await request.app.state.fetch_url(
            body.url,
            max_bytes=settings.max_upload_bytes,
            timeout_s=settings.url_fetch_timeout_s,
            allow_private=settings.allow_private_urls,
        )
    except FetchError as e:
        raise (BadRequest if e.blocked else UpstreamError)(str(e)) from None
    doc, job = await create_document(
        session,
        blobs,
        collection,
        data=fetched.data,
        filename=fetched.filename,
        content_type=fetched.content_type,
        source_type="url",
        source_uri=fetched.url,
        title=body.title or (None if fetched.filename else urlsplit(fetched.url).hostname),
    )
    request.app.state.worker.wake()
    return to_out(doc, job)


@router.get("", response_model=DocumentList)
async def list_documents(
    collection_id: uuid.UUID,
    access: ViewerAccess,
    session: SessionDep,
    status: DocumentStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    where = [Document.collection_id == collection.id]
    if status is not None:
        where.append(Document.status == status)
    total = (
        await session.execute(select(func.count()).select_from(Document).where(*where))
    ).scalar_one()
    rows = (
        (
            await session.execute(
                select(Document)
                .where(*where)
                .order_by(Document.created_at.desc(), Document.id)
                .limit(limit)
                .offset(offset)
            )
        )
        .scalars()
        .all()
    )
    jobs = await latest_jobs(session, [d.id for d in rows])
    return DocumentList(items=[to_out(d, jobs.get(d.id)) for d in rows], total=total)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document_detail(
    collection_id: uuid.UUID, document_id: uuid.UUID, access: ViewerAccess, session: SessionDep
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    return to_out(doc, (await latest_jobs(session, [doc.id])).get(doc.id))


class PageOut(BaseModel):
    page: int
    char_start: int
    char_end: int


class HeadingOut(BaseModel):
    level: int
    text: str
    char_start: int
    page: int | None


class ParsedOut(BaseModel):
    title: str | None
    text: str | None  # omitted with include_text=false
    characters: int
    pages: list[PageOut]
    headings: list[HeadingOut]
    metadata: dict[str, Any]
    warnings: list[str]


@router.get("/{document_id}/parsed", response_model=ParsedOut)
async def get_parsed(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    access: ViewerAccess,
    session: SessionDep,
    blobs: BlobsDep,
    include_text: bool = True,
):
    """The parsed text with its page map and headings: what chunking and citations index into."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    try:
        raw = await blobs.get(keys.parsed_key(collection.workspace_id, collection.id, doc.id))
    except BlobNotFound:
        raise NotFound("This document has not been parsed yet") from None
    parsed = ParsedDocument.from_dict(json.loads(raw))
    return ParsedOut(
        title=parsed.title,
        text=parsed.text if include_text else None,
        characters=len(parsed.text),
        pages=[
            PageOut(page=p.page, char_start=p.char_start, char_end=p.char_end) for p in parsed.pages
        ],
        headings=[
            HeadingOut(level=h.level, text=h.text, char_start=h.char_start, page=h.page)
            for h in parsed.headings
        ],
        metadata=parsed.metadata,
        warnings=parsed.warnings,
    )


@router.get("/{document_id}/original", response_class=Response)
async def download_original(
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    access: ViewerAccess,
    session: SessionDep,
    blobs: BlobsDep,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    try:
        data = await blobs.get(keys.original_key(collection.workspace_id, collection.id, doc.id))
    except BlobNotFound:
        log.error("original file missing for document %s", doc.id)
        raise NotFound("The original file is no longer available") from None
    name = (doc.meta or {}).get("filename") or f"{doc.id}"
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_")
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"
    return Response(
        data,
        # Always an attachment with a generic type: an uploaded .html must never render
        # in our origin.
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    request: Request,
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    access: MemberAccess,
    session: SessionDep,
    blobs: BlobsDep,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    prefix = keys.document_prefix(collection.workspace_id, collection.id, doc.id)
    # Derived indexes first: if the vector store is down this fails with a 502 and the document
    # is still there to retry, rather than leaving vectors that nothing refers to.
    await purge_document(request.app.state.providers, collection.id, doc.id)
    await session.delete(doc)
    await session.commit()
    # After the commit: a leftover blob is harmless, a row pointing at a missing blob is not.
    await blobs.delete_prefix(prefix)
    return Response(status_code=204)
