"""Concrete pipeline stages."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from sqlalchemy import delete

from app.db.models import Chunk
from app.ingest import keys
from app.ingest.chunking import ChunkingConfig, build_chunker
from app.ingest.chunking.contextual import ContextualChunker
from app.ingest.files import Kind, UnsupportedFile, detect_kind
from app.ingest.parsers import ParsedDocument, ParseError, parse
from app.ingest.pipeline import PermanentError, StageContext
from app.storage import BlobNotFound

log = logging.getLogger("atlas.stages")


async def parse_stage(ctx: StageContext) -> None:
    """Read the stored original, parse it and save the structured result next to it.

    Writes `parsed.json` (text, pages, headings, metadata) and a small summary on the document.
    Idempotent: running it again overwrites both.
    """
    doc, col = ctx.document, ctx.collection
    try:
        data = await ctx.blobs.get(keys.original_key(col.workspace_id, col.id, doc.id))
    except BlobNotFound:
        raise PermanentError("The original file is missing from storage") from None

    meta = doc.meta or {}
    try:
        kind = Kind(meta["kind"]) if meta.get("kind") else detect_kind(data, meta.get("filename"))
    except (ValueError, UnsupportedFile) as e:
        raise PermanentError(str(e)) from None

    try:
        # Parsing is CPU-bound; a thread keeps the event loop (API, SSE) responsive.
        parsed = await asyncio.to_thread(parse, data, kind, ocr=ctx.ocr)
    except ParseError as e:
        raise PermanentError(str(e)) from None

    payload = json.dumps(parsed.to_dict(), ensure_ascii=False).encode("utf-8")
    await ctx.blobs.put(keys.parsed_key(col.workspace_id, col.id, doc.id), payload)
    doc.meta = {**meta, "parse": parsed.summary()}  # new dict so the JSON column is saved
    log.info(
        "document parsed",
        extra={"document_id": str(doc.id), "kind": kind.value, "characters": len(parsed.text)},
    )


async def chunk_stage(ctx: StageContext) -> None:
    """Split the parsed text with the collection's strategy and store the chunks.

    Replaces any chunks the document already has, so re-running (a retry, or a re-index after a
    strategy change) is safe. Children get their parent's id; parents are stored but flagged as
    not indexable (`meta.kind == "parent"`).
    """
    doc, col = ctx.document, ctx.collection
    try:
        raw = await ctx.blobs.get(keys.parsed_key(col.workspace_id, col.id, doc.id))
    except BlobNotFound:
        raise PermanentError("The parsed text is missing; re-run the document") from None
    parsed = ParsedDocument.from_dict(json.loads(raw))

    cfg = ChunkingConfig(**(col.chunking_config or {}))
    chunker = build_chunker(col.chunking_strategy, ctx.providers)
    drafts = await chunker.chunk(parsed, cfg)
    if not drafts:
        raise PermanentError("No text could be split into chunks")

    await ctx.session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
    ids = [uuid.uuid4() for _ in drafts]
    for ordinal, d in enumerate(drafts):
        ctx.session.add(
            Chunk(
                id=ids[ordinal],
                document_id=doc.id,
                parent_id=ids[d.parent] if d.parent is not None else None,
                ordinal=ordinal,
                text=d.text,
                char_start=d.char_start,
                char_end=d.char_end,
                page=d.page,
                token_count=d.token_count,
                meta={
                    "kind": d.kind,
                    "heading_path": d.heading_path,
                    **({"context": d.context} if d.context else {}),
                },
            )
        )
    searchable = [d for d in drafts if d.indexable]
    summary = {
        "strategy": col.chunking_strategy.value,
        "chunks": len(searchable),
        "parents": len(drafts) - len(searchable),
        "avg_tokens": round(sum(d.token_count for d in searchable) / len(searchable)),
        "max_tokens": max(d.token_count for d in searchable),
    }
    if isinstance(chunker, ContextualChunker):
        summary["context"] = chunker.report.as_dict()
    doc.meta = {**(doc.meta or {}), "chunking": summary}
    log.info("document chunked", extra={"document_id": str(doc.id), **summary})
