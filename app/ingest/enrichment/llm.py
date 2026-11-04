"""LLM enrichment: a one-line summary and hypothetical questions for each chunk.

Why questions? Users ask questions, but documents contain statements. Embedding the questions a
chunk *answers* ("How long do I have to request a refund?") puts them close to the way people
search, which helps questions land on the right chunk even when they share few words with it.
The summary gives the chunk a compact, searchable gist.

One call per chunk returns both as JSON. Calls run with bounded concurrency, replies are
validated strictly (bad JSON or junk questions are dropped, never stored), and cost is capped by
`max_chunks`. An isolated failure just leaves that chunk without LLM metadata; a provider that
is clearly down raises so the job retries.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from app.ingest.enrichment.config import EnrichmentConfig
from app.providers.base import LLM, ChatMessage
from app.providers.errors import ProviderError

log = logging.getLogger("atlas.enrichment.llm")

MAX_CHUNK_CHARS = 6_000
MAX_SUMMARY_CHARS = 300
MAX_QUESTION_CHARS = 200
MIN_QUESTION_CHARS = 12
MAX_FAILURES_BEFORE_ABORT = 3

SYSTEM = (
    "You write search metadata for a document passage. Reply with a single JSON object and "
    "nothing else."
)
USER = (
    "Section: {path}\n\nPassage:\n<passage>\n{text}\n</passage>\n\n"
    "Return JSON with these keys:\n{spec}\n"
    "Use the passage's own language. Do not invent facts that are not in the passage."
)
SPEC_SUMMARY = '"summary": one sentence (max 30 words) stating what the passage says'
SPEC_QUESTIONS = (
    '"questions": a list of up to {n} different questions a user might ask that this passage '
    "directly answers; each must make sense on its own, without the passage"
)


@dataclass
class LlmEnrichment:
    summary: str | None = None
    questions: list[str] = field(default_factory=list)


@dataclass
class EnrichmentReport:
    calls: int = 0
    failed: int = 0
    invalid: int = 0  # replies that were not usable JSON
    skipped: int = 0  # chunks beyond the max_chunks cap
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def parse_reply(raw: str, cfg: EnrichmentConfig) -> LlmEnrichment | None:
    """Extract and validate the JSON object from a model reply. None if nothing usable."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out = LlmEnrichment()
    summary = data.get("summary")
    if cfg.summaries and isinstance(summary, str):
        summary = re.sub(r"\s+", " ", summary).strip()
        if summary:
            out.summary = summary[:MAX_SUMMARY_CHARS].rstrip()
    questions = data.get("questions")
    if cfg.questions and isinstance(questions, list):
        seen: set[str] = set()
        for q in questions:
            if not isinstance(q, str):
                continue
            q = re.sub(r"\s+", " ", q).strip()
            key = q.lower().rstrip("?").strip()
            if (
                MIN_QUESTION_CHARS <= len(q) <= MAX_QUESTION_CHARS
                and q.endswith(("?", "？"))
                and key not in seen
            ):
                seen.add(key)
                out.questions.append(q)
            if len(out.questions) == cfg.max_questions:
                break
    return out if (out.summary or out.questions) else None


def build_messages(path: list[str], text: str, cfg: EnrichmentConfig) -> list[ChatMessage]:
    spec = []
    if cfg.summaries:
        spec.append(SPEC_SUMMARY)
    if cfg.questions:
        spec.append(SPEC_QUESTIONS.format(n=cfg.max_questions))
    return [
        ChatMessage("system", SYSTEM),
        ChatMessage(
            "user",
            USER.format(
                path=" > ".join(path) or "(none)",
                text=text[:MAX_CHUNK_CHARS],
                spec="\n".join(f"- {s}" for s in spec),
            ),
        ),
    ]


async def enrich_with_llm(
    llm: LLM,
    items: list[tuple[list[str], str]],
    cfg: EnrichmentConfig,
) -> tuple[list[LlmEnrichment | None], EnrichmentReport]:
    """Enrich `(heading_path, text)` items; returns results in input order plus a usage report.

    Items beyond `cfg.max_chunks` are skipped (None). Raises ProviderError if the provider is
    clearly unavailable so the caller's job can be retried.
    """
    report = EnrichmentReport()
    results: list[LlmEnrichment | None] = [None] * len(items)
    if not cfg.uses_llm or not items:
        return results, report
    limit = min(len(items), cfg.max_chunks)
    report.skipped = len(items) - limit
    sem = asyncio.Semaphore(cfg.concurrency)
    successes = 0
    abort: ProviderError | None = None

    async def one(i: int) -> None:
        nonlocal successes, abort
        async with sem:
            if abort is not None:
                return
            path, text = items[i]
            report.calls += 1
            try:
                reply = await llm.complete(
                    build_messages(path, text, cfg), temperature=0.2, max_tokens=400
                )
            except ProviderError as e:
                report.failed += 1
                log.warning("enrichment call failed: %s", e)
                if report.failed >= MAX_FAILURES_BEFORE_ABORT and report.failed > successes:
                    abort = e
                return
            successes += 1
            report.prompt_tokens += reply.usage.prompt_tokens
            report.completion_tokens += reply.usage.completion_tokens
            parsed = parse_reply(reply.text, cfg)
            if parsed is None:
                report.invalid += 1
            results[i] = parsed

    await asyncio.gather(*(one(i) for i in range(limit)))
    if abort is not None:
        raise abort
    return results, report
