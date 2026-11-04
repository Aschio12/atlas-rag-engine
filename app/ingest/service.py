"""Adding and updating documents, independent of HTTP (Phase 2.1, reused by connectors in 2.7).

Everything that puts a file into a collection goes through here, so uploads, pasted text, URLs
and connectors share the same checks: the type is sniffed from the bytes, identical content is
deduplicated, the original is stored, the document row and its ingestion job are created in one
transaction, and a failure leaves no orphan files.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import Conflict, UnsupportedMedia
from app.db.models import Collection, Document, DocumentStatus, IngestJob
from app.ingest import keys, queue
from app.ingest.files import (
    UnsupportedFile,
    detect_kind,
    sanitize_filename,
    sha256_hex,
    title_from_filename,
)
from app.storage import BlobStore


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
    source_id: uuid.UUID | None = None,
) -> tuple[Document, IngestJob]:
    """Validate, deduplicate, store the original and insert the document row."""
    # Plain values up front: a rollback expires ORM objects and async sessions cannot lazy-load.
    collection_id, workspace_id = collection.id, collection.workspace_id
    try:
        kind = detect_kind(data, filename, content_type)
    except UnsupportedFile as e:
        raise UnsupportedMedia(str(e)) from None

    content_hash = sha256_hex(data)
    existing = await _find_by_hash(session, collection_id, content_hash)
    if existing is not None:
        raise _duplicate(existing.id, existing.title)

    safe_name = sanitize_filename(filename) if filename else None
    doc = Document(
        id=uuid.uuid4(),
        collection_id=collection_id,
        title=(title or "").strip() or title_from_filename(safe_name or "Untitled"),
        source_type=source_type,
        source_uri=source_uri,
        content_hash=content_hash,
        size_bytes=len(data),
        status=DocumentStatus.PENDING,
        source_id=source_id,
        meta={
            "filename": safe_name,
            "content_type": (content_type or "").split(";")[0].strip().lower() or None,
            "kind": kind.value,
            **(extra_meta or {}),
        },
    )
    folder = keys.document_prefix(workspace_id, collection_id, doc.id)
    await blobs.put(keys.original_key(workspace_id, collection_id, doc.id), data)
    try:
        session.add(doc)
        await session.flush()  # the document row must exist before its job references it
        job = await queue.enqueue(session, doc.id)
        await session.commit()
    except IntegrityError:
        # Lost a race with an identical upload: the unique (collection, hash) constraint held.
        await session.rollback()
        await blobs.delete_prefix(folder)
        row = await _find_by_hash(session, collection_id, content_hash)
        if row is None:
            raise
        raise _duplicate(row.id, row.title) from None
    except BaseException:
        await blobs.delete_prefix(folder)
        raise
    await session.refresh(doc)
    await session.refresh(job)
    return doc, job


async def _find_by_hash(session: AsyncSession, collection_id: uuid.UUID, content_hash: str):
    return (
        await session.execute(
            select(Document.id, Document.title).where(
                Document.collection_id == collection_id, Document.content_hash == content_hash
            )
        )
    ).first()


def _duplicate(existing_id: uuid.UUID, title: str) -> Conflict:
    return Conflict(
        f"This content is already in the collection as '{title}'",
        extra={"existing_document_id": str(existing_id)},
    )


async def find_by_source_uri(
    session: AsyncSession, collection_id: uuid.UUID, source_uri: str
) -> Document | None:
    """The document a connector already created for this address, if any."""
    return (
        await session.execute(
            select(Document).where(
                Document.collection_id == collection_id, Document.source_uri == source_uri
            )
        )
    ).scalar_one_or_none()


async def replace_content(
    session: AsyncSession,
    blobs: BlobStore,
    collection: Collection,
    doc: Document,
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    extra_meta: dict[str, Any] | None = None,
) -> IngestJob | None:
    """Update a document whose source changed (a re-synced page or file).

    Returns None when the content is identical (nothing to do), otherwise stores the new
    original, resets the document and queues a full re-parse. The old chunks and vectors stay
    searchable until the new ones are complete.
    """
    try:
        kind = detect_kind(data, filename, content_type)
    except UnsupportedFile as e:
        raise UnsupportedMedia(str(e)) from None
    content_hash = sha256_hex(data)
    if content_hash == doc.content_hash:
        return None
    clash = await _find_by_hash(session, collection.id, content_hash)
    if clash is not None and clash.id != doc.id:
        raise _duplicate(clash.id, clash.title)

    await blobs.put(keys.original_key(collection.workspace_id, collection.id, doc.id), data)
    safe_name = sanitize_filename(filename) if filename else None
    doc.content_hash = content_hash
    doc.size_bytes = len(data)
    doc.status = DocumentStatus.PENDING
    doc.meta = {
        **(doc.meta or {}),
        "filename": safe_name,
        "content_type": (content_type or "").split(";")[0].strip().lower() or None,
        "kind": kind.value,
        **(extra_meta or {}),
    }
    job = await queue.enqueue(session, doc.id, options={"reparse": True})
    await session.commit()
    await session.refresh(job)
    return job
