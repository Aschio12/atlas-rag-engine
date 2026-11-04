"""Removing everything derived from documents (Phase 2.8): vectors, files, and any extra index.

Order matters and is the same everywhere: derived indexes first, then the database rows, then
files. If the vector store is down the delete fails (the caller sees a 502 and the document is
still there to retry), so vectors are never orphaned by a half-finished delete. Orphaned *files*
are harmless and tolerated: they are removed last, after the commit.

Other indexes (the knowledge graph arrives in Phase 5) register cleanup hooks here instead of
every delete path having to know about them.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable

from app.ingest.vectors import vector_collection
from app.providers.registry import Providers

log = logging.getLogger("atlas.purge")

DocumentHook = Callable[[uuid.UUID, uuid.UUID], Awaitable[None]]  # (collection_id, document_id)
CollectionHook = Callable[[uuid.UUID], Awaitable[None]]  # (collection_id)

_document_hooks: list[DocumentHook] = []
_collection_hooks: list[CollectionHook] = []


def on_document_deleted(hook: DocumentHook) -> DocumentHook:
    """Register extra cleanup for a deleted document (e.g. graph nodes). Usable as a decorator."""
    if hook not in _document_hooks:
        _document_hooks.append(hook)
    return hook


def on_collection_deleted(hook: CollectionHook) -> CollectionHook:
    if hook not in _collection_hooks:
        _collection_hooks.append(hook)
    return hook


def clear_hooks() -> None:
    """For tests."""
    _document_hooks.clear()
    _collection_hooks.clear()


async def purge_document(providers: Providers, collection_id: uuid.UUID, document_id: uuid.UUID):
    """Delete a document's chunks' vectors (including question points) and run cleanup hooks."""
    await providers.vector_store.delete(
        vector_collection(collection_id), filters={"document_id": str(document_id)}
    )
    for hook in list(_document_hooks):
        await hook(collection_id, document_id)


async def purge_collection(providers: Providers, collection_id: uuid.UUID) -> None:
    """Drop a collection's whole vector collection and run cleanup hooks."""
    await providers.vector_store.drop_collection(vector_collection(collection_id))
    for hook in list(_collection_hooks):
        await hook(collection_id)


async def delete_documents(
    session,
    providers: Providers,
    blobs,
    collection,
    document_ids: list[uuid.UUID],
) -> int:
    """Delete several documents completely: vectors and hooks first, then the rows, then files.

    One document at a time, committing each, so an outage part-way leaves a consistent
    database (the remaining documents are simply still there)."""
    from sqlalchemy import delete

    from app.db.models import Document
    from app.ingest import keys

    removed = 0
    for doc_id in document_ids:
        await purge_document(providers, collection.id, doc_id)
        await session.execute(delete(Document).where(Document.id == doc_id))
        await session.commit()
        await blobs.delete_prefix(
            keys.document_prefix(collection.workspace_id, collection.id, doc_id)
        )
        removed += 1
    return removed
