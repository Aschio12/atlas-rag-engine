"""The retrieval pipeline: query in, ranked passages and a trace out (Phase 3).

    query -> transforms -> channels (dense, sparse; one pair per query variant)
          -> resolve against the database -> fuse -> rerank -> hits

Each stage records what it did in the trace, and every hit carries the score each stage gave it
and the reasons it is where it is.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection
from app.providers.errors import ProviderError
from app.providers.registry import Providers
from app.retrieve.candidates import Candidate, fetch_size, search_dense, search_sparse
from app.retrieve.filters import CompiledFilter
from app.retrieve.fusion import Fused, reciprocal_rank_fusion, weighted_fusion
from app.retrieve.hits import build_hit, matched_terms
from app.retrieve.options import RetrievalOptions
from app.retrieve.rerank import RerankUnavailable, rerank_hits
from app.retrieve.resolve import Row, chunk_id_of, load_rows, resolve
from app.retrieve.transform import Turn, Variant, transform_query
from app.retrieve.types import Hit, RetrievalResult, RetrievalTrace


@dataclass(frozen=True)
class Job:
    """One search: a kind of channel run with one piece of query text."""

    label: str  # "dense", "sparse", "dense[hyde]", "sparse[multi_query:2]"
    kind: str  # "dense" | "sparse"
    text: str
    variant: str | None = None  # None for the main query


@dataclass
class Channel:
    job: Job
    candidates: list[Candidate]  # servable, one per chunk, best first

    @property
    def label(self) -> str:
        return self.job.label

    @property
    def kind(self) -> str:
        return self.job.kind

    @property
    def by_chunk(self) -> dict[uuid.UUID, tuple[int, Candidate]]:
        return {chunk_id_of(c.point): (rank, c) for rank, c in enumerate(self.candidates, start=1)}


def make_jobs(primary: str, variants: list[Variant], mode: str) -> list[Job]:
    kinds = ["dense", "sparse"] if mode == "hybrid" else [mode]
    jobs = [Job(kind, kind, primary) for kind in kinds]
    for v in variants:
        for kind in v.kinds:
            if kind in kinds:
                jobs.append(Job(f"{kind}[{v.label}]", kind, v.text, v.label))
    return jobs


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
        history: list[Turn] | None = None,
    ) -> RetrievalResult:
        if any(c.workspace_id != workspace_id for c in collections):
            raise ValueError("every searched collection must belong to the workspace")
        options = options or RetrievalOptions()
        metadata = metadata or CompiledFilter()
        reranker = self._providers.reranker if options.rerank.enabled else None
        if options.rerank.enabled and reranker is None:
            raise RerankUnavailable("Reranking was requested but RERANK_PROVIDER is not set")
        trace = RetrievalTrace(query)
        warnings: list[str] = []

        started = time.perf_counter()
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

        primary, variants = query, []
        if options.transforms.any:
            primary, variants = await self._transform(
                query, history or [], options, trace, warnings
            )

        jobs = make_jobs(primary, variants, options.mode)
        channels, rows = await self._search(
            jobs, workspace_id, collections, metadata, options, trace
        )

        window = max(options.top_k, options.rerank.candidates) if reranker else options.top_k
        hits = self._rank(channels, rows, options, trace, window)
        if reranker is not None:
            hits = await self._rerank(reranker, primary, hits, trace, warnings)
        hits = hits[: options.top_k]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        trace.add("result", time.perf_counter(), returned=len(hits))
        return RetrievalResult(hits, trace, warnings)

    # -- stages ------------------------------------------------------------------------------

    async def _transform(self, query, history, options, trace, warnings):
        started = time.perf_counter()
        tr = await transform_query(
            self._providers.llm, query, history, options.transforms, mode=options.mode
        )
        warnings.extend(tr.warnings)
        trace.add(
            "transform",
            started,
            enabled=sorted(k for k, v in options.transforms.model_dump().items() if v is True),
            original=query,
            primary=tr.primary,
            variants=[
                {"label": v.label, "text": v.text, "channels": list(v.kinds)} for v in tr.variants
            ],
            skipped=tr.skipped,
            llm_calls=tr.llm_calls,
            tokens={"prompt": tr.usage.prompt_tokens, "completion": tr.usage.completion_tokens},
            took_ms=tr.took_ms,
        )
        return tr.primary, tr.variants

    async def _search(self, jobs, workspace_id, collections, metadata, options, trace):
        store = self._providers.vector_store
        limit = fetch_size(options.pool)
        models = sorted({c.embed_model for c in collections})

        vectors: dict[tuple[str, str], list[float]] = {}  # (model, text) -> vector
        dense_texts = list(dict.fromkeys(j.text for j in jobs if j.kind == "dense"))
        if dense_texts:
            started = time.perf_counter()
            for model in models:
                embedded = await self._providers.embedder_for(model).embed(dense_texts)
                vectors.update({(model, t): v for t, v in zip(dense_texts, embedded, strict=True)})
            trace.add("embed_query", started, models=models, texts=len(dense_texts))

        async def search(job: Job) -> list[Candidate]:
            if job.kind == "dense":
                per_model = {m: vectors[(m, job.text)] for m in models}
                return await search_dense(
                    store, collections, per_model, workspace_id, metadata, limit
                )
            return await search_sparse(store, collections, job.text, workspace_id, metadata, limit)

        started = time.perf_counter()
        found = await asyncio.gather(*(search(j) for j in jobs))
        # similarity is comparable across collections only for one model (dense) or one
        # collection (sparse: each has its own term statistics)
        comparable = {"dense": len(models) == 1, "sparse": len(collections) == 1}
        extra: dict[str, dict] = {}
        for job, candidates in zip(jobs, found, strict=True):
            details = {
                "points_found": len(candidates),
                "fetched_per_collection": limit,
                "comparable_scores": comparable[job.kind],
            }
            if job.variant is None:
                trace.add(f"{job.kind}_search", started, **details)
            else:
                extra[job.label] = {"query": job.text, **details}
        if extra:
            trace.add("variant_search", started, channels=extra)

        started = time.perf_counter()
        rows = await load_rows(
            self._session, {i for cs in found for c in cs if (i := chunk_id_of(c.point))}
        )
        channels: list[Channel] = []
        totals = {"missing": 0, "stale": 0, "folded_into_chunk": 0, "below_min_score": 0}
        per_channel: dict[str, dict[str, int]] = {}
        for job, candidates in zip(jobs, found, strict=True):
            kept, stats = resolve(candidates, rows)
            stats["below_min_score"] = 0
            if options.min_score is not None and job.kind == "dense":
                above = [c for c in kept if c.point.score >= options.min_score]
                stats["below_min_score"] = len(kept) - len(above)
                kept = above
            kept = kept[: options.pool]
            channels.append(Channel(job, kept))
            per_channel[job.label] = {**stats, "kept": len(kept)}
            for key in totals:
                totals[key] += stats[key]
        trace.add("resolve", started, channels=per_channel, **totals)
        return channels, rows

    def _rank(
        self,
        channels: list[Channel],
        rows: dict[uuid.UUID, Row],
        options: RetrievalOptions,
        trace: RetrievalTrace,
        window: int,
    ) -> list[Hit]:
        maps = {c.label: c.by_chunk for c in channels}
        fused: list[Fused] | None = None
        fusion = options.fusion
        if len(channels) > 1:
            started = time.perf_counter()
            weights = {
                c.label: fusion.weights()[c.kind]
                * (fusion.variant_weight if c.job.variant else 1.0)
                for c in channels
            }
            if fusion.method == "rrf":
                fused = reciprocal_rank_fusion(
                    {c.label: list(maps[c.label]) for c in channels}, k=fusion.k, weights=weights
                )
            else:
                scored = {
                    c.label: {i: cand.point.score for i, (_, cand) in maps[c.label].items()}
                    for c in channels
                }
                fused = weighted_fusion(scored, weights=weights)
            trace.add(
                "fusion",
                started,
                method=fusion.method,
                k=fusion.k if fusion.method == "rrf" else None,
                weights=weights,
                list_sizes={c.label: len(c.candidates) for c in channels},
                found_by_several_channels=sum(1 for f in fused if len(f.ranks) > 1),
                fused=len(fused),
            )
            order = [(f.key, f.score, f) for f in fused]
        else:
            (only,) = channels
            order = [(i, cand.point.score, None) for i, (_, cand) in only.by_chunk.items()]

        hits: list[Hit] = []
        for chunk_id, score, f in order[:window]:
            chunk, doc = rows[chunk_id]
            present = [(c, maps[c.label][chunk_id]) for c in channels if chunk_id in maps[c.label]]
            scores = {c.label: cand.point.score for c, (_, cand) in present}
            reasons = [_channel_reason(c, rank, cand, chunk) for c, (rank, cand) in present]
            if f is not None:
                scores[fusion.method] = f.score
                reasons.append(_fusion_reason(fusion.method, f))
            dense = [cand for c, (_, cand) in present if c.kind == "dense"]
            best = dense[0] if dense else present[0][1][1]
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

    async def _rerank(self, reranker, query, hits, trace, warnings) -> list[Hit]:
        started = time.perf_counter()
        try:
            outcome = await rerank_hits(reranker, self._providers.rerank_cache, query, hits)
        except ProviderError as exc:
            # Reranking sharpens the order but is not needed to answer: serve the fused order
            # and say so, rather than fail the whole search.
            warnings.append(f"Reranking failed ({exc}); results are in fused order")
            trace.add("rerank", started, failed=True, error=str(exc), candidates=len(hits))
            return hits
        trace.add("rerank", started, **outcome.details)
        return outcome.hits


def _channel_reason(channel: Channel, rank: int, cand: Candidate, chunk) -> str:
    via = f" for {channel.job.text[:80]!r}" if channel.job.variant else ""
    if channel.kind == "dense":
        return f"{channel.label}: cosine similarity {cand.point.score:.4f} (rank {rank}){via}"
    shared = matched_terms(channel.job.text, chunk)
    terms_note = f"; matched terms: {', '.join(shared)}" if shared else ""
    return f"{channel.label}: BM25 score {cand.point.score:.4f} (rank {rank}){via}{terms_note}"


def _fusion_reason(method: str, f: Fused) -> str:
    ranks = ", ".join(f"{name} #{rank}" for name, rank in f.ranks.items())
    label = "reciprocal rank fusion" if method == "rrf" else "weighted fusion"
    return f"{label}: {ranks} -> {f.score:.4f}"
