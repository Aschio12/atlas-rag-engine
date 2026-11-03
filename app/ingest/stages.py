"""Concrete pipeline stages."""

from __future__ import annotations

import asyncio
import json
import logging

from app.ingest import keys
from app.ingest.files import Kind, UnsupportedFile, detect_kind
from app.ingest.parsers import ParseError, parse
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
