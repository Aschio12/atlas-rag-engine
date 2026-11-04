"""Concrete pipeline stages."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from sqlalchemy import delete, select

from app.db.models import Chunk
from app.ingest import keys
from app.ingest.chunking import ChunkingConfig, build_chunker
from app.ingest.chunking.contextual import ContextualChunker
from app.ingest.enrichment.config import EnrichmentConfig
from app.ingest.enrichment.keywords import extract_keywords
from app.ingest.enrichment.language import detect_language, dominant_language
from app.ingest.enrichment.llm import enrich_with_llm
from app.ingest.files import Kind, UnsupportedFile, detect_kind
from app.ingest.parsers import ParsedDocument, ParseError, parse
from app.ingest.pipeline import PermanentError, StageContext
from app.ingest.sparse import encode_document
from app.ingest.vectors import (
    base_payload,
    embedding_text,
    lexical_text,
    question_point_id,
    vector_collection,
)
from app.providers.base import Point
from app.providers.errors import ProviderError
from app.storage import BlobNotFound

log = logging.getLogger("atlas.stages")


async def parse_stage(ctx: StageContext) -> None:
    """Read the stored original, parse it and save the structured result next to it.

    Writes `parsed.json` (text, pages, headings, metadata) and a small summary on the document.
    Idempotent: running it again overwrites both.
    """
    doc, col = ctx.document, ctx.collection
    if ctx.options.get("reparse") is False and await ctx.blobs.exists(
        keys.parsed_key(col.workspace_id, col.id, doc.id)
    ):
        # A re-index with unchanged text: reuse the stored parse instead of re-running OCR/PDF work.
        log.info("reusing stored parse", extra={"document_id": str(doc.id)})
        return
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
        parsed = await asyncio.to_thread(
            parse, data, kind, ocr=ctx.ocr, filename=meta.get("filename")
        )
    except ParseError as e:
        raise PermanentError(str(e)) from None

    if meta.get("title_provisional") and parsed.title:
        doc.title = parsed.title[:500]  # a crawled page or repo doc is titled by its own title
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


async def enrich_stage(ctx: StageContext) -> None:
    """Add metadata to each searchable chunk: language and keywords (offline), and optionally an
    LLM summary and hypothetical questions. Stored in the chunk's metadata; idempotent, and
    fields that are switched off (or were skipped by a cap) are removed rather than left stale.
    """
    doc, col = ctx.document, ctx.collection
    cfg = EnrichmentConfig(**(col.enrichment_config or {}))
    rows = (
        (
            await ctx.session.execute(
                select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.ordinal)
            )
        )
        .scalars()
        .all()
    )
    chunks = [c for c in rows if (c.meta or {}).get("kind", "chunk") != "parent"]
    texts = [c.text for c in chunks]

    languages = [detect_language(t) if cfg.language else None for t in texts]
    keywords = extract_keywords(texts, cfg.max_keywords) if cfg.keywords else [[] for _ in texts]
    await ctx.progress(0.2)
    llm_results, report = await enrich_with_llm(
        ctx.providers.llm,
        [((c.meta or {}).get("heading_path", []), c.text) for c in chunks],
        cfg,
    )

    questions = 0
    for chunk, lang, kws, extra in zip(chunks, languages, keywords, llm_results, strict=True):
        meta = {
            k: v
            for k, v in (chunk.meta or {}).items()
            if k not in ("language", "keywords", "summary", "questions")
        }
        if lang:
            meta["language"] = lang.code
        if kws:
            meta["keywords"] = kws
        if extra and extra.summary:
            meta["summary"] = extra.summary
        if extra and extra.questions:
            meta["questions"] = extra.questions
            questions += len(extra.questions)
        chunk.meta = meta  # a new dict, so the JSON column is saved

    language = dominant_language([c.meta.get("language") for c in chunks])
    summary = {
        "language": language,
        "keywords": cfg.keywords,
        "summaries": sum(1 for c in chunks if c.meta.get("summary")),
        "questions": questions,
    }
    if cfg.uses_llm:
        summary["llm"] = report.as_dict()
    doc.meta = {**(doc.meta or {}), "language": language, "enrichment": summary}
    log.info(
        "document enriched",
        extra={"document_id": str(doc.id), **{k: v for k, v in summary.items() if k != "llm"}},
    )


async def _searchable_chunks(ctx: StageContext) -> list[Chunk]:
    rows = (
        (
            await ctx.session.execute(
                select(Chunk).where(Chunk.document_id == ctx.document.id).order_by(Chunk.ordinal)
            )
        )
        .scalars()
        .all()
    )
    return [c for c in rows if (c.meta or {}).get("kind", "chunk") != "parent"]


async def embed_stage(ctx: StageContext) -> None:
    """Embed every searchable chunk (and each hypothetical question) and write dense + sparse
    vectors to the vector store, batch by batch.

    All points of one run share a fresh `index_version`. Nothing is deleted here: the indexing
    stage removes the previous run's points only after the new ones are all in place, so a
    document never disappears from search while it is being re-indexed. Idempotent: point ids
    are deterministic, so a retry overwrites instead of duplicating.
    """
    doc, col, settings = ctx.document, ctx.collection, ctx.settings
    chunks = await _searchable_chunks(ctx)
    embedder = ctx.providers.embedder_for(col.embed_model)
    store = ctx.providers.vector_store
    name = vector_collection(col.id)
    version = uuid.uuid4().hex

    # (point id, payload, text to embed, text for the lexical index)
    items: list[tuple[str, dict, str, str]] = []
    for chunk in chunks:
        payload = base_payload(col, doc, chunk, version)
        items.append(
            (
                str(chunk.id),
                {**payload, "kind": "chunk"},
                embedding_text(chunk),
                lexical_text(chunk),
            )
        )
        for i, question in enumerate((chunk.meta or {}).get("questions", [])):
            items.append(
                (
                    question_point_id(chunk.id, i),
                    {**payload, "kind": "question", "question": question},
                    question,
                    question,
                )
            )

    truncated = 0
    dimension: int | None = None
    batch = settings.embed_batch_size
    for start in range(0, len(items), batch):
        part = items[start : start + batch]
        texts = []
        for _, _, text, _ in part:
            if len(text) > settings.embed_max_chars:
                text, truncated = text[: settings.embed_max_chars], truncated + 1
            texts.append(text)
        vectors = await embedder.embed(texts)
        if len(vectors) != len(part):
            raise PermanentError("The embedding model returned the wrong number of vectors")
        if dimension is None:
            dimension = len(vectors[0])
            try:
                await store.ensure_collection(name, dimension, sparse=True)
            except ValueError:
                raise PermanentError(
                    "The collection's vectors have a different size than the embedding model "
                    "produces; re-index the collection"
                ) from None
        await store.upsert(
            name,
            [
                Point(pid, vec, payload, encode_document(lexical))
                for (pid, payload, _, lexical), vec in zip(part, vectors, strict=True)
            ],
        )
        await ctx.progress(min(start + batch, len(items)) / len(items))

    doc.meta = {
        **(doc.meta or {}),
        "indexing": {
            "version": version,
            "points": len(items),
            "chunks": len(chunks),
            "questions": len(items) - len(chunks),
            "dimension": dimension,
            "model": col.embed_model,
            "truncated": truncated,
        },
    }
    log.info("document embedded", extra={"document_id": str(doc.id), "points": len(items)})


async def index_stage(ctx: StageContext) -> None:
    """Finish an indexing run: verify that every point of this run is present, then remove the
    document's points from earlier runs (re-chunking, a changed strategy, edited content)."""
    doc, col = ctx.document, ctx.collection
    info = (doc.meta or {}).get("indexing") or {}
    version = info.get("version")
    if not version:
        raise PermanentError("The document was not embedded; re-run it")
    store = ctx.providers.vector_store
    name = vector_collection(col.id)

    present = await store.count(name, {"document_id": str(doc.id), "index_version": version})
    if present != info["points"]:
        # The store lost writes or is lagging: retry the whole run rather than serve a hole.
        raise ProviderError(store.name, f"expected {info['points']} points, found {present}")
    await store.delete(
        name, filters={"document_id": str(doc.id)}, exclude={"index_version": version}
    )
    doc.meta = {**(doc.meta or {}), "indexing": {**info, "verified": present}}
    log.info("document indexed", extra={"document_id": str(doc.id), "points": present})
