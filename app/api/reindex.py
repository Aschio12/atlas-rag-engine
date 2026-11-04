"""Re-indexing and index status (Phase 2.8).

A collection's chunking strategy, chunk sizes, enrichment and embedding model are fixed when it
is created because they decide what is stored. Changing any of them is done here, explicitly:
the new settings are saved and every document is queued to be rebuilt (re-chunked, re-enriched,
re-embedded) from its stored text. Old vectors stay searchable until each document's new ones
are complete (see `index_version`), except when the embedding model changes: vectors from two
models cannot share an index, so that index is dropped first and search for the collection is
partial until the jobs finish. Watch `GET .../index`.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

from app.api.collections import CollectionOut, to_out
from app.api.documents import (
    DocumentOut,
    get_collection,
    get_document,
)
from app.api.documents import (
    to_out as document_out,
)
from app.core.auth import AdminAccess, MemberAccess, ViewerAccess
from app.core.deps import SessionDep
from app.core.errors import Conflict
from app.db.models import ChunkingStrategy, Document, DocumentStatus, IngestJob
from app.ingest import queue
from app.ingest.chunking import ChunkingConfig
from app.ingest.enrichment.config import EnrichmentConfig
from app.ingest.purge import purge_collection
from app.ingest.vectors import vector_collection

router = APIRouter(prefix="/workspaces/{workspace_id}/collections/{collection_id}", tags=["index"])


class DocumentReindex(BaseModel):
    reparse: bool = Field(
        default=False, description="Parse the original file again instead of reusing its text"
    )


class CollectionReindex(BaseModel):
    chunking_strategy: ChunkingStrategy | None = None
    chunking_config: ChunkingConfig | None = None
    enrichment: EnrichmentConfig | None = None
    embed_model: str | None = Field(default=None, min_length=1, max_length=200)
    reparse: bool = False


class ReindexOut(BaseModel):
    queued: int
    collection: CollectionOut


class IndexStatus(BaseModel):
    strategy: ChunkingStrategy
    embed_model: str
    documents: dict[str, int]
    active_jobs: int
    searchable_chunks: int
    expected_points: int
    vector_points: int
    in_sync: bool


async def _active_jobs(session, collection_id: uuid.UUID) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(IngestJob)
            .join(Document, Document.id == IngestJob.document_id)
            .where(
                Document.collection_id == collection_id,
                IngestJob.state.in_(queue.ACTIVE_STATES),
            )
        )
    ).scalar_one()


@router.post("/documents/{document_id}/reindex", response_model=DocumentOut)
async def reindex_document(
    document_id: uuid.UUID,
    request: Request,
    access: MemberAccess,
    session: SessionDep,
    collection_id: uuid.UUID,
    body: DocumentReindex | None = None,
):
    """Rebuild one document's chunks, enrichment and vectors with the collection's settings."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    doc = await get_document(session, collection, document_id)
    if await _doc_active(session, doc):
        raise Conflict("The document is still being processed; wait for it to finish")
    reparse = body.reparse if body else False
    await session.execute(
        update(Document).where(Document.id == doc.id).values(status=DocumentStatus.PENDING)
    )
    job = await queue.enqueue(session, doc.id, options={"reparse": reparse})
    await session.commit()
    request.app.state.worker.wake()
    await session.refresh(doc)
    await session.refresh(job)
    return document_out(doc, job)


async def _doc_active(session, doc: Document) -> bool:
    return (
        await session.execute(
            select(func.count())
            .select_from(IngestJob)
            .where(IngestJob.document_id == doc.id, IngestJob.state.in_(queue.ACTIVE_STATES))
        )
    ).scalar_one() > 0


@router.post("/reindex", response_model=ReindexOut)
async def reindex_collection(
    request: Request,
    access: AdminAccess,
    session: SessionDep,
    collection_id: uuid.UUID,
    body: CollectionReindex | None = None,
):
    """Apply new chunking / enrichment / embedding settings (optional) and rebuild every
    document. Refused while any document is still processing, so no document ends up half old
    and half new."""
    body = body or CollectionReindex()
    collection = await get_collection(session, access.workspace.id, collection_id)
    busy = await _active_jobs(session, collection.id)
    if busy:
        raise Conflict(f"{busy} document(s) are still processing; try again when they finish")

    # Request bodies are validated by pydantic; an omitted config keeps the current one.
    new_config = body.chunking_config or ChunkingConfig(**(collection.chunking_config or {}))
    embed_changed = bool(body.embed_model and body.embed_model != collection.embed_model)

    if embed_changed:
        await purge_collection(request.app.state.providers, collection.id)  # 502 if index down
    if body.chunking_strategy is not None:
        collection.chunking_strategy = body.chunking_strategy
    collection.chunking_config = new_config.model_dump()
    if body.enrichment is not None:
        collection.enrichment_config = body.enrichment.model_dump()
    if embed_changed:
        collection.embed_model = body.embed_model

    docs = (
        (await session.execute(select(Document.id).where(Document.collection_id == collection.id)))
        .scalars()
        .all()
    )
    for doc_id in docs:
        await queue.enqueue(session, doc_id, options={"reparse": body.reparse})
    if docs:
        await session.execute(
            update(Document)
            .where(Document.collection_id == collection.id)
            .values(status=DocumentStatus.PENDING)
        )
    await session.commit()
    request.app.state.worker.wake()
    await session.refresh(collection)
    return ReindexOut(queued=len(docs), collection=to_out(collection, len(docs)))


@router.get("/index", response_model=IndexStatus)
async def index_status(
    request: Request, access: ViewerAccess, session: SessionDep, collection_id: uuid.UUID
):
    """How indexed a collection is: documents by status, and the vector store's point count
    against what the documents say they wrote. `in_sync` means nothing is processing and the
    two agree."""
    collection = await get_collection(session, access.workspace.id, collection_id)
    rows = (
        await session.execute(
            select(Document.status, Document.meta).where(Document.collection_id == collection.id)
        )
    ).all()
    by_status = {s.value: 0 for s in DocumentStatus}
    expected = chunks = 0
    for status, meta in rows:
        by_status[status.value] += 1
        if status == DocumentStatus.READY:
            meta = meta or {}
            expected += (meta.get("indexing") or {}).get("points", 0)
            chunks += (meta.get("chunking") or {}).get("chunks", 0)
    active = await _active_jobs(session, collection.id)
    points = await request.app.state.providers.vector_store.count(vector_collection(collection.id))
    return IndexStatus(
        strategy=collection.chunking_strategy,
        embed_model=collection.embed_model,
        documents={**by_status, "total": len(rows)},
        active_jobs=active,
        searchable_chunks=chunks,
        expected_points=expected,
        vector_points=points,
        in_sync=active == 0 and points == expected,
    )
