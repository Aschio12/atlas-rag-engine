"""How chunks map to vector-store points.

Each searchable chunk is one point whose id *is* the chunk id. Each hypothetical question of a
chunk (Phase 2.5) is an extra point with a derived, stable id, so a query that resembles a
question lands on the chunk that answers it (`chunk_id` in the payload leads back).

The text itself is not stored in the vector store: the database is the single source of truth
and retrieval hydrates hits from it. Payloads carry only what filtering and citation need.
"""

from __future__ import annotations

import uuid

from app.db.models import Chunk, Collection, Document

QUESTION_NAMESPACE = uuid.UUID("6f1c3a52-8f2e-4d0b-a3c4-5b7d9e0f1a2b")


def vector_collection(collection_id: uuid.UUID) -> str:
    """One vector collection per Atlas collection (each has its own embedding model/dimension)."""
    return f"atlas_{collection_id.hex}"


def question_point_id(chunk_id: uuid.UUID, index: int) -> str:
    return str(uuid.uuid5(QUESTION_NAMESPACE, f"{chunk_id}:{index}"))


def embedding_text(chunk: Chunk) -> str:
    """What is embedded for a chunk: its situating context (contextual chunking), if any, then
    the passage."""
    context = (chunk.meta or {}).get("context")
    return f"{context}\n\n{chunk.text}" if context else chunk.text


def lexical_text(chunk: Chunk) -> str:
    """What the sparse index sees: section names, context, summary, keywords and the passage,
    so a search for a heading or a keyword finds the chunk even if the passage never says it."""
    meta = chunk.meta or {}
    parts = [
        " ".join(meta.get("heading_path", [])),
        meta.get("context") or "",
        meta.get("summary") or "",
        " ".join(meta.get("keywords", [])),
        chunk.text,
    ]
    return "\n".join(p for p in parts if p)


def base_payload(collection: Collection, document: Document, chunk: Chunk, version: str) -> dict:
    meta = chunk.meta or {}
    return {
        "workspace_id": str(collection.workspace_id),
        "collection_id": str(collection.id),
        "document_id": str(document.id),
        "chunk_id": str(chunk.id),
        "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
        "ordinal": chunk.ordinal,
        "page": chunk.page,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "heading_path": meta.get("heading_path", []),
        "language": meta.get("language"),
        "title": document.title,
        "index_version": version,
    }
