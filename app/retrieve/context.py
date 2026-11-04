"""Context assembly (Phase 3.6): from ranked passages to the context an answer is written from.

Retrieval returns the passages that *match*; the model needs passages that *help*, within a
token budget. The steps, in order:

1. **Dedupe.** Drop near-copies of a better-ranked passage (boilerplate repeated across
   documents, the same text in two versions). Similarity is word-shingle Jaccard.
2. **MMR.** Maximal marginal relevance reorders the candidates so each next pick is relevant but
   not redundant with what is already chosen: `lambda * relevance - (1 - lambda) * max
   similarity to the picks`. Similarity uses the stored embeddings (same collection) and falls
   back to word shingles.
3. **Expand.** A child chunk of a parent-child collection is replaced by its parent section;
   otherwise `neighbors` chunks either side can be added.
4. **Merge.** Pieces of one document that overlap or touch are merged into one block, and so are
   pieces separated only by whitespace (chunkers trim it): the stored document text is read and
   the block is that exact slice, so nothing is repeated and nothing invented. A gap that skips
   real content keeps two blocks apart.
5. **Pack.** Blocks are added in MMR order while they fit the token budget. A block that does not
   fit is retried without its expansion, then skipped; later, smaller blocks can still fit.
6. **Order.** `edges` places the strongest blocks at the start and end of the context, since
   models attend least to the middle of a long prompt; `relevance` and `document` order are also
   available.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk
from app.ingest import keys
from app.ingest.chunking.base import estimate_tokens
from app.ingest.vectors import vector_collection
from app.providers.base import VectorStore
from app.providers.errors import ProviderError
from app.retrieve.options import ContextOptions
from app.retrieve.types import ContextBlock, ContextResult, Hit
from app.storage import BlobNotFound, BlobStore

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


# ---- pure helpers ---------------------------------------------------------------------------


def shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    if len(words) <= size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def dedupe(hits: list[Hit], threshold: float) -> tuple[list[Hit], list[dict]]:
    """Keep each hit unless it is a near-copy of a better-ranked one that was kept."""
    kept: list[tuple[Hit, set]] = []
    dropped: list[dict] = []
    for hit in hits:
        sig = shingles(hit.text)
        twin = next(((k, s) for k, s in kept if jaccard(sig, s) >= threshold), None)
        if twin is None:
            kept.append((hit, sig))
        else:
            dropped.append(
                {
                    "chunk_id": str(hit.chunk_id),
                    "duplicate_of": str(twin[0].chunk_id),
                    "similarity": round(jaccard(sig, twin[1]), 4),
                }
            )
    return [h for h, _ in kept], dropped


def mmr_order(
    keys: Sequence,
    relevance: dict,
    similarity: Callable[[object, object], float],
    lam: float,
) -> list:
    """Order `keys` by maximal marginal relevance. Relevance is min-max normalised first so it
    is on the same 0..1 scale as similarity. Ties keep the incoming order."""
    if not keys:
        return []
    values = [relevance[k] for k in keys]
    low, high = min(values), max(values)
    norm = {k: ((relevance[k] - low) / (high - low) if high > low else 1.0) for k in keys}
    remaining, chosen = list(keys), []
    while remaining:

        def score(k):
            redundancy = max((similarity(k, c) for c in chosen), default=0.0)
            return lam * norm[k] - (1 - lam) * redundancy

        best = max(remaining, key=score)  # max() keeps the first of equals
        remaining.remove(best)
        chosen.append(best)
    return chosen


def edge_order(by_relevance: list) -> list:
    """Best first, second best last, third second, ...: the weakest end up in the middle."""
    front, back = [], []
    for i, item in enumerate(by_relevance):
        (front if i % 2 == 0 else back).append(item)
    return front + back[::-1]


@dataclass(frozen=True)
class Piece:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    start: int
    end: int
    text: str
    ordinal: int


@dataclass
class Block:
    document_id: uuid.UUID
    start: int
    end: int
    text: str
    pieces: list[Piece]
    retrieved: list[uuid.UUID]
    score: float
    first: Hit  # supplies title, page and heading
    order: int  # position in MMR order of the best hit inside
    notes: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


MAX_GAP = 4  # characters of whitespace between chunks that still count as adjacent


class Joiner:
    """Decides whether two spans of a document can be one block, and builds the text.

    Overlapping or touching spans always can. A short gap can too, but only when the document
    text is available to prove the gap is whitespace; then the block is the exact slice."""

    def __init__(self, texts: dict[uuid.UUID, str] | None = None):
        self.texts = texts or {}

    def joinable(self, doc: uuid.UUID, end: int, start: int) -> bool:
        if start <= end:
            return True
        text = self.texts.get(doc)
        return text is not None and start - end <= MAX_GAP and not text[end:start].strip()

    def bridge(self, doc: uuid.UUID, end: int, start: int) -> str:
        return self.texts[doc][end:start] if start > end else ""


def coalesce(
    pieces: list[Piece], joiner: Joiner | None = None
) -> list[tuple[int, int, str, list[Piece]]]:
    """Merge pieces of one document that can be joined into runs of exact text."""
    joiner = joiner or Joiner()
    runs: list[tuple[int, int, str, list[Piece]]] = []
    for piece in sorted(pieces, key=lambda p: (p.start, p.end, str(p.chunk_id))):
        if runs and joiner.joinable(piece.document_id, runs[-1][1], piece.start):
            start, end, text, members = runs[-1]
            if piece.end > end:
                text += joiner.bridge(piece.document_id, end, piece.start)
                text += piece.text[max(end - piece.start, 0) :]
                end = piece.end
            if piece.chunk_id not in {m.chunk_id for m in members}:
                members = [*members, piece]
            runs[-1] = (start, end, text, members)
        else:
            runs.append((piece.start, piece.end, piece.text, [piece]))
    return runs


def _blocks_for(
    pieces: list[Piece], hit: Hit, order: int, notes: list[str], joiner: Joiner | None = None
) -> list[Block]:
    return [
        Block(
            document_id=hit.document_id,
            start=start,
            end=end,
            text=text,
            pieces=members,
            # the run that contains the hit's own span is where the retrieved chunk sits (a parent
            # section or a merged neighbourhood contains it without being it)
            retrieved=[hit.chunk_id] if start <= hit.char_start and hit.char_end <= end else [],
            score=hit.score,
            first=hit,
            order=order,
            notes=list(notes),
        )
        for start, end, text, members in coalesce(pieces, joiner)
    ]


def add_blocks(
    existing: list[Block], new: list[Block], joiner: Joiner | None = None
) -> list[Block]:
    """Add blocks, merging any that join a block of the same document."""
    joiner = joiner or Joiner()
    result = list(existing)
    for block in new:
        touching = [
            b
            for b in result
            if b.document_id == block.document_id
            and joiner.joinable(b.document_id, b.end, block.start)
            and joiner.joinable(b.document_id, block.end, b.start)
        ]
        if not touching:
            result.append(block)
            continue
        group = [*touching, block]
        pieces = [p for b in group for p in b.pieces]
        ((start, end, text, members),) = coalesce(pieces, joiner)
        best = min(group, key=lambda b: b.order)
        merged = Block(
            document_id=block.document_id,
            start=start,
            end=end,
            text=text,
            pieces=members,
            retrieved=list(dict.fromkeys(i for b in group for i in b.retrieved)),
            score=max(b.score for b in group),
            first=min((b.first for b in group), key=lambda h: h.char_start),
            order=best.order,
            notes=[
                *dict.fromkeys(n for b in group for n in b.notes),
                "merged overlapping passages",
            ],
        )
        result = [b for b in result if all(b is not t for t in touching)]
        result.append(merged)
    return result


# ---- the stage ------------------------------------------------------------------------------


def _piece(hit: Hit) -> Piece:
    return Piece(hit.chunk_id, hit.document_id, hit.char_start, hit.char_end, hit.text, hit.ordinal)


def _piece_of_row(c: Chunk) -> Piece:
    return Piece(c.id, c.document_id, c.char_start, c.char_end, c.text, c.ordinal)


class ContextAssembler:
    def __init__(self, session: AsyncSession, store: VectorStore, blobs: BlobStore | None = None):
        self._session = session
        self._store = store
        self._blobs = blobs  # without it only overlapping or touching pieces merge

    async def assemble(
        self, candidates: list[Hit], options: ContextOptions
    ) -> tuple[ContextResult, dict]:
        pool = candidates[: options.pool]
        details: dict = {"pool": len(pool), "budget": options.max_tokens}
        hits = pool
        if options.dedupe:
            hits, dropped = dedupe(pool, options.dedupe_threshold)
            details["deduplicated"] = dropped
        if options.mmr and len(hits) > 1:
            hits, details["mmr"] = await self._mmr(hits, options)

        parents, neighbors = await self._related(hits, options)
        joiner = await self._joiner(hits, parents, neighbors)
        blocks: list[Block] = []
        skipped: list[dict] = []
        for order, hit in enumerate(hits):
            base_pieces = [_piece(hit)]
            notes: list[str] = []
            pieces = base_pieces
            if options.parent_expansion and hit.parent_id in parents:
                pieces = [parents[hit.parent_id]]
                notes.append("expanded to its parent section")
            elif options.neighbors:
                extra = [
                    neighbors[(hit.document_id, hit.ordinal + d)]
                    for d in range(-options.neighbors, options.neighbors + 1)
                    if d != 0 and (hit.document_id, hit.ordinal + d) in neighbors
                ]
                if extra:
                    pieces = [*extra, *base_pieces]
                    notes.append(f"added {len(extra)} neighbouring chunk(s)")
            attempts = [(pieces, notes)]
            if pieces is not base_pieces:  # if the expanded passage does not fit, try the chunk
                attempts.append((base_pieces, ["too large to expand within the budget"]))
            for use, why in attempts:
                candidate = add_blocks(blocks, _blocks_for(use, hit, order, why, joiner), joiner)
                if sum(b.tokens for b in candidate) <= options.max_tokens:
                    blocks = candidate
                    break
            else:
                skipped.append({"chunk_id": str(hit.chunk_id), "reason": "does not fit the budget"})
        details["skipped"] = skipped

        ranked = sorted(blocks, key=lambda b: (-b.score, b.order))
        if options.order == "edges":
            ordered = edge_order(ranked)
        elif options.order == "document":
            first_rank: dict[uuid.UUID, int] = {}
            for i, b in enumerate(ranked):
                first_rank.setdefault(b.document_id, i)
            ordered = sorted(blocks, key=lambda b: (first_rank[b.document_id], b.start))
        else:
            ordered = ranked
        out = [self._to_block(f"S{i}", b) for i, b in enumerate(ordered, start=1)]
        total = sum(b.tokens for b in out)
        details.update(tokens=total, blocks=len(out), order=options.order)
        return ContextResult(out, total, options.max_tokens), details

    # -- steps -------------------------------------------------------------------------------

    async def _mmr(self, hits: list[Hit], options: ContextOptions) -> tuple[list[Hit], dict]:
        vectors: dict[uuid.UUID, list[float]] = {}
        failed = False
        by_collection: dict[uuid.UUID, list[Hit]] = defaultdict(list)
        for h in hits:
            by_collection[h.collection_id].append(h)
        for collection_id, group in by_collection.items():
            try:
                found = await self._store.get_vectors(
                    vector_collection(collection_id), [str(h.chunk_id) for h in group]
                )
            except ProviderError:
                failed = True
                continue
            vectors.update({uuid.UUID(k): v for k, v in found.items()})
        sigs = {h.chunk_id: shingles(h.text) for h in hits}
        by_id = {h.chunk_id: h for h in hits}
        used = {"embeddings": 0, "text": 0}

        def similarity(a: uuid.UUID, b: uuid.UUID) -> float:
            va, vb = vectors.get(a), vectors.get(b)
            same_space = by_id[a].collection_id == by_id[b].collection_id
            if va and vb and same_space and len(va) == len(vb):
                used["embeddings"] += 1
                return max(cosine(va, vb), 0.0)
            used["text"] += 1
            return jaccard(sigs[a], sigs[b])

        order = mmr_order(
            [h.chunk_id for h in hits],
            {h.chunk_id: h.score for h in hits},
            similarity,
            options.mmr_lambda,
        )
        reordered = [by_id[i] for i in order]
        moved = [
            {"chunk_id": str(h.chunk_id), "from": hits.index(h) + 1, "to": i}
            for i, h in enumerate(reordered, start=1)
            if hits.index(h) + 1 != i
        ]
        details = {
            "lambda": options.mmr_lambda,
            "similarity_from": "embeddings" if used["embeddings"] >= used["text"] else "text",
            "moved": moved,
        }
        if failed:
            details["note"] = "the vector store could not supply embeddings; compared text instead"
        return reordered, details

    async def _related(
        self, hits: list[Hit], options: ContextOptions
    ) -> tuple[dict[uuid.UUID, Piece], dict[tuple[uuid.UUID, int], Piece]]:
        parents: dict[uuid.UUID, Piece] = {}
        if options.parent_expansion:
            ids = {h.parent_id for h in hits if h.parent_id}
            if ids:
                rows = (
                    (await self._session.execute(select(Chunk).where(Chunk.id.in_(ids))))
                    .scalars()
                    .all()
                )
                parents = {c.id: _piece_of_row(c) for c in rows}
        neighbors: dict[tuple[uuid.UUID, int], Piece] = {}
        if options.neighbors:
            wanted: dict[uuid.UUID, set[int]] = defaultdict(set)
            for h in hits:
                if h.parent_id in parents and options.parent_expansion:
                    continue  # the parent section already includes its siblings
                for d in range(-options.neighbors, options.neighbors + 1):
                    if d and h.ordinal + d >= 0:
                        wanted[h.document_id].add(h.ordinal + d)
            for doc_id, ordinals in wanted.items():
                rows = (
                    (
                        await self._session.execute(
                            select(Chunk).where(
                                Chunk.document_id == doc_id, Chunk.ordinal.in_(ordinals)
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                for c in rows:
                    if (c.meta or {}).get("kind", "chunk") != "parent":  # only searchable chunks
                        neighbors[(doc_id, c.ordinal)] = _piece_of_row(c)
        return parents, neighbors

    async def _joiner(self, hits, parents, neighbors) -> Joiner:
        """Load the text of the documents where two candidate pieces are a short gap apart."""
        if self._blobs is None:
            return Joiner()
        pieces: dict[uuid.UUID, list[Piece]] = defaultdict(list)
        for h in hits:
            pieces[h.document_id].append(_piece(h))
        for p in [*parents.values(), *neighbors.values()]:
            pieces[p.document_id].append(p)
        need: set[uuid.UUID] = set()
        for doc_id, items in pieces.items():
            items.sort(key=lambda p: p.start)
            for a, b in zip(items, items[1:], strict=False):
                if 0 < b.start - max(x.end for x in items if x.start <= a.start) <= MAX_GAP:
                    need.add(doc_id)
        by_doc = {h.document_id: h for h in hits}
        texts: dict[uuid.UUID, str] = {}
        for doc_id in need:
            hit = by_doc.get(doc_id)
            if hit is None:
                continue
            workspace_id = await self._workspace_of(hit.collection_id)
            try:
                raw = await self._blobs.get(
                    keys.parsed_key(workspace_id, hit.collection_id, doc_id)
                )
                texts[doc_id] = json.loads(raw)["text"]
            except (BlobNotFound, KeyError, ValueError):
                continue  # no text, no gap merging for this document
        return Joiner(texts)

    async def _workspace_of(self, collection_id: uuid.UUID) -> uuid.UUID:
        from app.db.models import Collection

        return (
            await self._session.execute(
                select(Collection.workspace_id).where(Collection.id == collection_id)
            )
        ).scalar_one()

    @staticmethod
    def _to_block(label: str, b: Block) -> ContextBlock:
        return ContextBlock(
            label=label,
            text=b.text,
            document_id=b.document_id,
            collection_id=b.first.collection_id,
            title=b.first.title,
            page=b.first.page,
            heading_path=b.first.heading_path,
            char_start=b.start,
            char_end=b.end,
            chunk_ids=[p.chunk_id for p in sorted(b.pieces, key=lambda p: p.start)],
            retrieved_chunk_ids=list(b.retrieved),
            score=b.score,
            tokens=b.tokens,
            notes=b.notes,
        )
