"""The retrieval pipeline: query in, ranked passages and a trace out (Phase 3).

    query -> channels (dense, sparse) -> resolve against the database -> fuse -> hits

Each stage records what it did in the trace, and every hit carries the score each stage gave it
and the reasons it is where it is. Later stages (reranking, query transforms, context assembly)
plug into this same flow.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection
from app.providers.registry import Providers
from app.retrieve.candidates import Candidate, fetch_size, search_dense, search_sparse
from app.retrieve.filters import CompiledFilter
from app.retrieve.fusion import Fused, reciprocal_rank_fusion, weighted_fusion
from app.retrieve.hits import build_hit, matched_terms
from app.retrieve.options import RetrievalOptions
from app.retrieve.resolve import Row, chunk_id_of, load_rows, resolve
from app.retrieve.types import Hit, RetrievalResult, RetrievalTrace


@dataclass
class Channel:
    label: str  # "dense", "sparse"
    kind: str  # "dense" | "sparse"
    text: str  # the query text this channel searched with
    candidates: list[Candidate]  # servable, one per chunk, best first

    @property
    def by_chunk(self) -> dict[uuid.UUID, tuple[int, Candidate]]:
        return {chunk_id_of(c.point): (rank, c) for rank, c in enumerate(self.candidates, start=1)}


class RetrievalPipeline:
    def __init__(self, session: AsyncSession, providers: Providers):
        self._session = session
        self._providers = providers

    async def run(
        self,
        query: str,
        workspace_id: uuid.UUID,
        collections: list[Collection],
        options: RetrievalOptions | None = None,
        metadata: CompiledFilter | None = None,
    ) -> RetrievalResult:
        if any(c.workspace_id != workspace_id for c in collections):
            raise ValueError("every searched collection must belong to the workspace")
        options = options or RetrievalOptions()
        metadata = metadata or CompiledFilter()
        trace = RetrievalTrace(query)

        started = time.perf_counter()
        models = sorted({c.embed_model for c in collections})
        trace.add(
            "scope",
            started,
            workspace_id=str(workspace_id),
            mode=options.mode,
            collections=[
                {"id": str(c.id), "name": c.name, "model": c.embed_model} for c in collections
            ],
            filters=list(metadata.summary),
        )
        if not collections:
            return RetrievalResult([], trace)

        kinds = ["dense", "sparse"] if options.mode == "hybrid" else [options.mode]
        limit = fetch_size(options.pool)
        store = self._providers.vector_store

        vectors: dict[str, list[float]] = {}
        if "dense" in kinds:
            started = time.perf_counter()
            for model in models:
                (vectors[model],) = await self._providers.embedder_for(model).embed([query])
            trace.add("embed_query", started, models=models)

        async def search(kind: str) -> list[Candidate]:
            if kind == "dense":
                return await search_dense(
                    store, collections, vectors, workspace_id, metadata, limit
                )
            return await search_sparse(store, collections, query, workspace_id, metadata, limit)

        started = time.perf_counter()
        found = await asyncio.gather(*(search(k) for k in kinds))
        for kind, candidates in zip(kinds, found, strict=True):
            trace.add(
                f"{kind}_search",
                started,
                points_found=len(candidates),
                fetched_per_collection=limit,
                # similarity is comparable across collections only for one model (dense) or
                # one collection (sparse: each has its own term statistics)
                comparable_scores=len(models) == 1 if kind == "dense" else len(collections) == 1,
            )

        started = time.perf_counter()
        rows = await load_rows(
            self._session, {i for cs in found for c in cs if (i := chunk_id_of(c.point))}
        )
        channels: list[Channel] = []
        totals = {"missing": 0, "stale": 0, "folded_into_chunk": 0, "below_min_score": 0}
        per_channel: dict[str, dict[str, int]] = {}
        for kind, candidates in zip(kinds, found, strict=True):
            kept, stats = resolve(candidates, rows)
            stats["below_min_score"] = 0
            if options.min_score is not None:
                above = [c for c in kept if c.point.score >= options.min_score]
                stats["below_min_score"] = len(kept) - len(above)
                kept = above
            kept = kept[: options.pool]
            channels.append(Channel(kind, kind, query, kept))
            per_channel[kind] = {**stats, "kept": len(kept)}
            for key in totals:
                totals[key] += stats[key]
        trace.add("resolve", started, channels=per_channel, **totals)

        hits = self._rank(channels, rows, options, trace)
        trace.add("result", time.perf_counter(), returned=len(hits))
        return RetrievalResult(hits, trace)

    # -- ranking -----------------------------------------------------------------------------

    def _rank(
        self,
        channels: list[Channel],
        rows: dict[uuid.UUID, Row],
        options: RetrievalOptions,
        trace: RetrievalTrace,
    ) -> list[Hit]:
        maps = {c.label: c.by_chunk for c in channels}
        fused: list[Fused] | None = None
        if len(channels) > 1:
            started = time.perf_counter()
            weights = {c.label: options.fusion.weights()[c.kind] for c in channels}
            if options.fusion.method == "rrf":
                fused = reciprocal_rank_fusion(
                    {c.label: list(maps[c.label]) for c in channels},
                    k=options.fusion.k,
                    weights=weights,
                )
            else:
                fused = weighted_fusion(
                    {
                        c.label: {i: cand.point.score for i, (_, cand) in maps[c.label].items()}
                        for c in channels
                    },
                    weights=weights,
                )
            both = sum(1 for f in fused if len(f.ranks) > 1)
            trace.add(
                "fusion",
                started,
                method=options.fusion.method,
                k=options.fusion.k if options.fusion.method == "rrf" else None,
                weights=weights,
                list_sizes={c.label: len(c.candidates) for c in channels},
                found_by_several_channels=both,
                fused=len(fused),
            )
            order = [(f.key, f.score, f) for f in fused]
        else:
            (only,) = channels
            order = [(i, cand.point.score, None) for i, (_, cand) in only.by_chunk.items()]

        hits: list[Hit] = []
        for chunk_id, score, f in order[: options.top_k]:
            chunk, doc = rows[chunk_id]
            present = [(c, maps[c.label][chunk_id]) for c in channels if chunk_id in maps[c.label]]
            scores = {c.label: cand.point.score for c, (_, cand) in present}
            reasons = [_channel_reason(c, rank, cand, chunk) for c, (rank, cand) in present]
            if f is not None:
                scores[options.fusion.method] = f.score
                reasons.append(_fusion_reason(options, f))
            best = (
                next(cand for c, (_, cand) in present if c.kind == "dense")
                if any(c.kind == "dense" for c, _ in present)
                else present[0][1][1]
            )
            hits.append(
                build_hit(
                    chunk,
                    doc,
                    rank=len(hits) + 1,
                    score=score,
                    best=best,
                    scores=scores,
                    reasons=reasons,
                )
            )
        return hits


def _channel_reason(channel: Channel, rank: int, cand: Candidate, chunk) -> str:
    if channel.kind == "dense":
        return f"dense: cosine similarity {cand.point.score:.4f} (rank {rank})"
    shared = matched_terms(channel.text, chunk)
    terms_note = f"; matched terms: {', '.join(shared)}" if shared else ""
    return f"sparse: BM25 score {cand.point.score:.4f} (rank {rank}){terms_note}"


def _fusion_reason(options: RetrievalOptions, f: Fused) -> str:
    ranks = ", ".join(f"{name} #{rank}" for name, rank in f.ranks.items())
    label = "reciprocal rank fusion" if options.fusion.method == "rrf" else "weighted fusion"
    return f"{label}: {ranks} -> {f.score:.4f}"
