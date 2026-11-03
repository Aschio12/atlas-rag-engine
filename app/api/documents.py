"""Documents: upload, list, inspect, download and delete (Phase 2.1)."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Query, Response, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import MemberAccess, ViewerAccess
from app.core.deps import BlobsDep, SessionDep, SettingsDep
from app.core.errors import Conflict, NotFound, PayloadTooLarge, UnsupportedMedia
from app.db.models import Collection, Document, DocumentStatus
from app.ingest import keys
from app.ingest.files import (
    UnsupportedFile,
    detect_kind,
    sanitize_filename,
    sha256_hex,
    title_from_filename,
)
from app.storage import BlobNotFound, BlobStore

log = logging.getLogger("atlas.documents")

router = APIRouter(
    prefix="/workspaces/{workspace_id}/collections/{collection_id}/documents",
    tags=["documents"],
)

MAX_TEXT_CHARS = 5_000_000
READ_CHUNK = 1024 * 1024


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


def to_out(d: Document) -> DocumentOut:
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


async def read_limited(file: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing to buffer more than `limit` bytes."""
    buf = bytearray()
    while chunk := await file.read(READ_CHUNK):
        buf += chunk
        if len(buf) > limit:
            raise PayloadTooLarge(f"The file is larger than the {limit // (1024 * 1024)} MB limit")
    return bytes(buf)


async def create_document(
    session: AsyncSession,
    blobs: BlobStore,
    collection: Collection,
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    source_type: str,
    source_uri: str | None = None,
    title: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Document:
    """Validate, deduplicate, store the original and insert the document row."""
    try:
        kind = detect_kind(data, filename, content_type)
    except UnsupportedFile as e:
        raise UnsupportedMedia(str(e)) from None

    content_hash = sha256_hex(data)
    existing = (
        await session.execute(
            select(Document.id, Document.title).where(
                Document.collection_id == collection.id, Document.content_hash == content_hash
            )
        )
    ).first()
    if existing is not None:
        raise _duplicate(existing.id, existing.title)

    safe_name = sanitize_filename(filename) if filename else None
    doc = Document(
        id=uuid.uuid4(),
        collection_id=collection.id,
        title=(title or "").strip() or title_from_filename(safe_name or "Untitled"),
        source_type=source_type,
        source_uri=source_uri,
        content_hash=content_hash,
        size_bytes=len(data),
        status=DocumentStatus.PENDING,
        meta={
            "filename": safe_name,
            "content_type": (content_type or "").split(";")[0].strip().lower() or None,
            "kind": kind.value,
            **(extra_meta or {}),
        },
    )
    key = keys.original_key(collection.workspace_id, collection.id, doc.id)
    await blobs.put(key, data)
    session.add(doc)
    try:
        await session.commit()
    except IntegrityError:
        # Lost a race with an identical upload: the unique (collection, hash) constraint held.
        await session.rollback()
        await blobs.delete(key)
        row = (
            await session.execute(
                select(Document.id, Document.title).where(
                    Document.collection_id == collection.id, Document.content_hash == content_hash
                )
            )
        ).first()
        if row is None:
            raise
        raise _duplicate(row.id, row.title) from None
    except BaseException:
        await blobs.delete(key)
        raise
    await session.refresh(doc)
    return doc


def _duplicate(existing_id: uuid.UUID, title: str) -> Conflict:
    return Conflict(
        f"This content is already in the collection as '{title}'",
        extra={"existing_document_id": str(existing_id)},
    )


@router.post("", status_code=201, response_model=DocumentOut)
async def upload_document(
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
    doc = await create_document(
        session,
        blobs,
        collection,
        data=data,
        filename=file.filename,
        content_type=file.content_type,
        source_type="file",
        title=title,
    )
    return to_out(doc)


@router.post("/text", status_code=201, response_model=DocumentOut)
async def create_text_document(
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
    doc = await create_document(
        session,
        blobs,
        collection,
        data=data,
        filename=f"{body.title}.{ext}",
        content_type=None,
        source_type="text",
        title=body.title,
    )
    return to_out(doc)


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
    return DocumentList(items=[to_out(d) for d in rows], total=total)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document_detail(
    collection_id: uuid.UUID, document_id: uuid.UUID, access: ViewerAccess, session: SessionDep
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    return to_out(await get_document(session, collection, document_id))


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
    collection_id: uuid.UUID,
    document_id: uuid.UUID,
    access: MemberAccess,
    session: SessionDep,
    blobs: BlobsDep,
):
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    prefix = keys.document_prefix(collection.workspace_id, collection.id, doc.id)
    await session.delete(doc)
    await session.commit()
    # After the commit: a leftover blob is harmless, a row pointing at a missing blob is not.
    await blobs.delete_prefix(prefix)
    return Response(status_code=204)
