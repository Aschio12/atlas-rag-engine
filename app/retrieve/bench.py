"""Retrieval benchmark (Phase 3.9): how good is each way of searching, measured on questions with
known answers.

A *golden set* is a list of questions, each with the evidence that answers it: a document and a
phrase that must appear in a retrieved passage. Judging by phrase rather than by chunk id means
the numbers survive a change of chunking strategy or chunk size, which is exactly what you will
want to compare next.

    {"id": "p01", "tag": "paraphrase", "question": "How long do I have to ask for my money back?",
     "evidence": [{"document": "billing", "contains": "request a refund within 30 days"}]}

A returned passage is *relevant* if it satisfies any evidence item. Per question and
configuration we compute recall@k (the share of evidence items found in the top k), hit@k, MRR and
nDCG@k (see `metrics`), then average. The standard configurations compare vector-only (dense),
BM25-only (sparse), hybrid, and hybrid with reranking.

Run it against the committed corpus with

    python -m app.retrieve.bench --out benchmarks/phase3/report.md

By default it uses the deterministic fake providers, which checks the wiring and gives a
regression baseline but says nothing about how real embedding models behave; pass `--use-env` to
use the providers configured in the environment (e.g. EMBED_PROVIDER=ollama, RERANK_PROVIDER=...).
The full evaluation harness (Phase 6) builds on the same idea with larger, generated sets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.retrieve.metrics import hit_at_k, mean, ndcg_at_k, percentile, reciprocal_rank
from app.retrieve.options import RerankOptions, RetrievalOptions
from app.retrieve.pipeline import RetrievalPipeline
from app.retrieve.types import Hit

DEFAULT_KS = (1, 3, 5, 10)


# ---- golden set -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    document: str  # the document's name (its title, ignoring case and punctuation)
    contains: str  # a phrase a retrieved passage must contain (ignoring case and spacing)


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    evidence: tuple[Evidence, ...]
    tag: str = ""


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _squash(text: str) -> str:
    return " ".join(text.casefold().split())


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            evidence = tuple(Evidence(e["document"], e["contains"]) for e in raw["evidence"])
            if not evidence or not raw["question"].strip():
                raise ValueError("a question and at least one evidence item are required")
            items.append(GoldenItem(raw["id"], raw["question"], evidence, raw.get("tag", "")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: question ids must be unique")
    return items


def satisfies(hit: Hit, evidence: Evidence) -> bool:
    return _normalise(hit.title) == _normalise(evidence.document) and _squash(
        evidence.contains
    ) in _squash(hit.text)


def judge(hits: list[Hit], item: GoldenItem) -> tuple[list[bool], list[set[int]]]:
    """For each hit, whether it is relevant, and which evidence items each hit satisfies."""
    satisfied = [{i for i, e in enumerate(item.evidence) if satisfies(h, e)} for h in hits]
    return [bool(s) for s in satisfied], satisfied


# ---- running ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    name: str
    options: RetrievalOptions


def standard_configs(top_k: int = max(DEFAULT_KS), rerank: bool = True) -> list[Config]:
    def make(**kw: Any) -> RetrievalOptions:
        return RetrievalOptions(top_k=top_k, **kw)

    configs = [
        Config("dense", make(mode="dense")),
        Config("sparse", make(mode="sparse")),
        Config("hybrid", make(mode="hybrid")),
    ]
    if rerank:
        configs.append(
            Config("hybrid+rerank", make(mode="hybrid", rerank=RerankOptions(enabled=True)))
        )
    return configs


@dataclass
class QueryResult:
    id: str
    tag: str
    question: str
    first_relevant_rank: int | None
    evidence_found: int
    evidence_total: int
    metrics: dict[str, float]
    latency_ms: float


@dataclass
class ConfigReport:
    name: str
    metrics: dict[str, float]
    by_tag: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    queries: list[QueryResult] = field(default_factory=list)


@dataclass
class Report:
    meta: dict[str, Any]
    configs: list[ConfigReport]
    ks: tuple[int, ...] = DEFAULT_KS

    def to_dict(self, detail: bool = False) -> dict[str, Any]:
        def round_all(d: dict[str, float]) -> dict[str, float]:
            return {k: round(v, 4) for k, v in d.items()}

        out: dict[str, Any] = {"meta": self.meta, "configs": {}}
        for c in self.configs:
            entry: dict[str, Any] = {
                "metrics": round_all(c.metrics),
                "by_tag": {t: round_all(m) for t, m in c.by_tag.items()},
                "latency_ms": round_all(c.latency_ms),
            }
            if detail:
                entry["queries"] = [
                    {
                        "id": q.id,
                        "tag": q.tag,
                        "first_relevant_rank": q.first_relevant_rank,
                        "evidence": f"{q.evidence_found}/{q.evidence_total}",
                    }
                    for q in c.queries
                ]
            out["configs"][c.name] = entry
        return out

    def to_markdown(self) -> str:
        ks = self.ks
        head = ["config", *[f"R@{k}" for k in ks], "MRR", f"nDCG@{ks[-1]}", "Hit@1", "p95 ms"]
        lines = ["# Retrieval benchmark", ""]
        for key, value in self.meta.items():
            lines.append(f"- **{key}**: {value}")
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for c in self.configs:
            m = c.metrics
            cells = [
                *[f"{m[f'recall@{k}']:.3f}" for k in ks],
                f"{m['mrr']:.3f}",
                f"{m[f'ndcg@{ks[-1]}']:.3f}",
                f"{m['hit@1']:.3f}",
                f"{c.latency_ms['p95']:.0f}",
            ]
            lines.append("| " + " | ".join([f"**{c.name}**", *cells]) + " |")
        tags = sorted({t for c in self.configs for t in c.by_tag})
        if tags:
            lines += ["", "## By question type (MRR / recall@5)", ""]
            lines += ["| type | " + " | ".join(c.name for c in self.configs) + " |"]
            lines += ["|---|" + "---|" * len(self.configs)]
            for tag in tags:
                row = []
                for c in self.configs:
                    m = c.by_tag.get(tag)
                    row.append(f"{m['mrr']:.2f} / {m['recall@5']:.2f}" if m else "-")
                lines.append(f"| {tag} | " + " | ".join(row) + " |")
        best = max(self.configs, key=lambda c: c.metrics["mrr"], default=None)
        if best is not None:
            misses = [
                q
                for q in best.queries
                if q.first_relevant_rank in (None,) or q.first_relevant_rank > 5
            ]
            lines += ["", f"## Questions `{best.name}` did not answer in the top 5", ""]
            lines += [f"- `{q.id}` ({q.tag}): {q.question}" for q in misses] or ["- none"]
        return "\n".join(lines) + "\n"


async def run_benchmark(
    pipeline: RetrievalPipeline,
    workspace_id: uuid.UUID,
    collections: list,
    items: list[GoldenItem],
    configs: list[Config],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    meta: dict[str, Any] | None = None,
) -> Report:
    """Search every question with every configuration and score the results."""
    if not items:
        raise ValueError("the golden set is empty")
    reports: list[ConfigReport] = []
    for config in configs:
        queries: list[QueryResult] = []
        for item in items:
            started = time.perf_counter()
            result = await pipeline.run(item.question, workspace_id, collections, config.options)
            elapsed = (time.perf_counter() - started) * 1000
            queries.append(_score(item, result.hits, ks, elapsed))
        reports.append(
            ConfigReport(
                name=config.name,
                metrics=_average([q.metrics for q in queries]),
                by_tag={
                    tag: _average([q.metrics for q in queries if q.tag == tag])
                    for tag in sorted({q.tag for q in queries if q.tag})
                },
                latency_ms={
                    "mean": mean([q.latency_ms for q in queries]),
                    "p95": percentile([q.latency_ms for q in queries], 95),
                },
                queries=queries,
            )
        )
    return Report({"questions": len(items), **(meta or {})}, reports, ks)


def _score(item: GoldenItem, hits: list[Hit], ks: tuple[int, ...], elapsed: float) -> QueryResult:
    relevance, satisfied = judge(hits, item)
    total = len(item.evidence)
    metrics: dict[str, float] = {"mrr": reciprocal_rank(relevance)}
    for k in ks:
        found = set().union(*satisfied[:k]) if satisfied[:k] else set()
        metrics[f"recall@{k}"] = len(found) / total
        metrics[f"hit@{k}"] = hit_at_k(relevance, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevance, total, k)
    first = next((i for i, r in enumerate(relevance, start=1) if r), None)
    return QueryResult(
        item.id,
        item.tag,
        item.question,
        first,
        len(set().union(*satisfied)) if satisfied else 0,
        total,
        metrics,
        elapsed,
    )


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean([r[key] for r in rows]) for key in rows[0]} if rows else {}


# ---- the command line -------------------------------------------------------------------------

CHUNKING = {"target_tokens": 128, "overlap_tokens": 16, "max_tokens": 256}
PASSWORD = "benchmark-pass-1"


async def build_and_run(
    corpus: Path, golden: Path, *, use_env: bool = False, settings: Any = None
) -> Report:
    """Index the corpus into a scratch workspace, then benchmark the standard configurations."""
    from httpx import ASGITransport, AsyncClient

    from app.core.config import Settings
    from app.db.base import Base
    from app.main import create_app

    storage = tempfile.mkdtemp(prefix="atlas-bench-")
    overrides = {
        "app_env": "test",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "storage_dir": storage,
        "worker_enabled": False,
        "ocr_enabled": False,
        "log_level": "WARNING",
        "jwt_secret": "benchmark-" + "x" * 40,
    }
    if settings is None:
        if use_env:
            settings = Settings(**overrides)
        else:
            fakes = {
                "llm_provider": "fake",
                "embed_provider": "fake",
                "rerank_provider": "fake",
                "vector_store": "memory",
            }
            settings = Settings(_env_file=None, **overrides, **fakes)
    app = create_app(settings)
    async with app.state.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://bench") as c:
            body = {"email": "bench@example.com", "password": PASSWORD}
            await c.post("/v1/auth/register", json=body)
            token = (await c.post("/v1/auth/login", json=body)).json()["access_token"]
            h = {"Authorization": f"Bearer {token}"}
            ws = (await c.post("/v1/workspaces", json={"name": "bench"}, headers=h)).json()["id"]
            col = (
                await c.post(
                    f"/v1/workspaces/{ws}/collections",
                    json={"name": "corpus", "chunking_config": CHUNKING},
                    headers=h,
                )
            ).json()["id"]
            files = sorted(corpus.glob("*.md"))
            if not files:
                raise ValueError(f"no .md files in {corpus}")
            for path in files:
                r = await c.post(
                    f"/v1/workspaces/{ws}/collections/{col}/documents",
                    files={"file": (path.name, path.read_bytes(), "text/markdown")},
                    headers=h,
                )
                r.raise_for_status()
            while await app.state.worker.run_once():
                pass
        from app.db.models import Collection, Document, DocumentStatus

        async with app.state.sessionmaker() as session:
            from sqlalchemy import select

            collections = list((await session.execute(select(Collection))).scalars())
            docs = list((await session.execute(select(Document))).scalars())
            not_ready = [d.title for d in docs if d.status != DocumentStatus.READY]
            if not_ready:
                raise RuntimeError(f"documents failed to index: {', '.join(not_ready)}")
            providers = app.state.providers
            pipeline = RetrievalPipeline(session, providers, app.state.blobs)
            rerank_on = providers.reranker is not None
            meta = {
                "corpus": f"{len(docs)} documents from {corpus.name}",
                "chunking": "recursive, ~128 tokens, 16 overlap",
                "embedding": f"{providers.embedder.name}:{providers.embedder.model}",
                "reranker": (
                    f"{providers.reranker.name}:{providers.reranker.model}" if rerank_on else "none"
                ),
                "vector store": providers.vector_store.name,
            }
            return await run_benchmark(
                pipeline,
                collections[0].workspace_id,
                collections,
                load_golden(golden),
                standard_configs(rerank=rerank_on),
                meta=meta,
            )
    finally:
        await app.state.engine.dispose()


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parents[2] / "benchmarks" / "phase3"
    parser = argparse.ArgumentParser(description="Benchmark Atlas retrieval on a golden set")
    parser.add_argument("--corpus", type=Path, default=here / "corpus")
    parser.add_argument("--golden", type=Path, default=here / "golden.jsonl")
    parser.add_argument("--out", type=Path, help="write the Markdown report here")
    parser.add_argument("--json", type=Path, help="write the metrics (with per-question ranks)")
    parser.add_argument(
        "--use-env",
        action="store_true",
        help="use the providers configured in the environment instead of the fake ones",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(build_and_run(args.corpus, args.golden, use_env=args.use_env))
    text = report.to_markdown()
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(detail=True), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
