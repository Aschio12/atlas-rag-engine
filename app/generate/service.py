"""Grounded answering (Phase 4): question in, a stream of events out, ending in a checked answer.

    policy check -> retrieve -> guardrails -> confidence -> prompt -> generate -> ground -> checks

Every stage can end the answer early, always with an explanation instead of a guess:

* a question that breaks a workspace content rule is refused before anything is retrieved;
* passages that look like prompt injections are flagged to the model (or dropped, by policy);
* when nothing relevant was found the assistant says so, without calling the model;
* a model that says NOT_FOUND, or leaks its hidden reference code, is handled, not shown.

The answer text is then validated against the context it was written from (`citations`), blended
into a confidence badge (`confidence`), and checked against the workspace's rules. The service
does not touch the database except to read: persisting a conversation turn is the caller's job.
"""

from __future__ import annotations

import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Collection
from app.generate import confidence as conf
from app.generate import prompts
from app.generate.citations import Grounding, ground
from app.generate.events import (
    NOT_FOUND,
    CitationScanner,
    Event,
    PrefixGate,
    split_answer,
)
from app.generate.injection import screen, summary
from app.generate.options import AnswerOptions
from app.generate.pii import Redactor
from app.generate.policy import WorkspacePolicy, stricter_injection, stricter_pii, violated_term
from app.generate.sources import render_sources, user_message
from app.providers.base import ChatMessage, Usage
from app.providers.errors import ProviderError
from app.providers.registry import Providers
from app.retrieve.options import ContextOptions, RerankOptions, RetrievalOptions
from app.retrieve.pipeline import RetrievalPipeline
from app.retrieve.rerank import RerankUnavailable
from app.retrieve.scope import resolve_collections
from app.retrieve.transform import Turn
from app.retrieve.types import ContextBlock, Hit
from app.storage import BlobStore

log = logging.getLogger("atlas.answer")

NOT_FOUND_MESSAGE = "I couldn't find this in your documents."
NO_ANSWER_MESSAGE = "I couldn't produce an answer. Please try again."
LEAK_MESSAGE = "I can't share that."
RELATED_SOURCES = 3


@dataclass
class Prepared:
    """A request after validation: everything resolved, nothing started."""

    collections: list[Collection]
    retrieval: RetrievalOptions
    pii: str
    injection: str
    abstain_below: float
    prompt: prompts.Prompt
    options: AnswerOptions


class AnswerError(Exception):
    """An error event, turned back into an exception for callers that want one."""

    def __init__(self, data: dict[str, Any]):
        super().__init__(data.get("message", "answering failed"))
        self.data = data


def _passage(p) -> dict[str, Any]:
    return {"chunk_id": str(p.chunk_id), "char_start": p.char_start, "char_end": p.char_end}


def source_dict(
    block: ContextBlock, flagged: str | None = None, text: str | None = None
) -> dict[str, Any]:
    out = {
        "label": block.label,
        "title": block.title,
        "page": block.page,
        "heading_path": block.heading_path,
        "document_id": str(block.document_id),
        "collection_id": str(block.collection_id),
        "char_start": block.char_start,
        "char_end": block.char_end,
        "chunk_ids": [str(i) for i in block.chunk_ids],
        "passages": [_passage(p) for p in block.passages],
        "score": block.score,
        "tokens": block.tokens,
        "text": block.text if text is None else text,
    }
    if flagged:
        out["flagged"] = flagged
    return out


def hit_dict(hit: Hit) -> dict[str, Any]:
    return {
        "chunk_id": str(hit.chunk_id),
        "document_id": str(hit.document_id),
        "title": hit.title,
        "page": hit.page,
        "rank": hit.rank,
        "score": hit.score,
        "scores": hit.scores,
    }


def citation_dict(c) -> dict[str, Any]:
    return {
        "label": c.label,
        "document_id": str(c.document_id),
        "collection_id": str(c.collection_id),
        "title": c.title,
        "page": c.page,
        "heading_path": c.heading_path,
        "char_start": c.char_start,
        "char_end": c.char_end,
        "chunk_ids": [str(i) for i in c.chunk_ids],
        "passages": [_passage(p) for p in c.passages],
        "mentions": c.mentions,
        "support": c.support,
        "supported": c.supported,
    }


def grounding_dict(g: Grounding | None, redact=None) -> dict[str, Any] | None:
    """The grounding report; `redact` (a function on text) cleans the sentences it quotes."""
    if g is None:
        return None
    clean = redact or (lambda text: text)
    return {
        "coverage": g.coverage,
        "supported_share": g.supported_share,
        "score": g.score,
        "uncited": [clean(t) for t in g.uncited],
        "invalid": g.invalid,
        "weak": g.weak,
        "claims": [
            {"text": clean(c.text), "labels": c.labels, "support": c.support} for c in g.claims
        ],
    }


def confidence_dict(c: conf.Confidence) -> dict[str, Any]:
    return {
        "score": round(c.score, 4),
        "level": c.level,
        "signals": c.signals,
        "reasons": c.reasons,
    }


class AnswerService:
    def __init__(self, providers: Providers, blobs: BlobStore | None, settings: Settings):
        self._providers = providers
        self._blobs = blobs
        self._settings = settings

    # -- validation ---------------------------------------------------------------------------

    async def prepare(
        self,
        session: AsyncSession,
        workspace_id: uuid.UUID,
        options: AnswerOptions,
        policy: WorkspacePolicy,
        *,
        has_history: bool = False,
    ) -> Prepared:
        """Resolve and validate everything that can fail before work starts (unknown
        collections, a prompt version that does not exist, reranking that is not configured)."""
        collections = await resolve_collections(session, workspace_id, options.collection_ids)
        version = options.prompt_version or self._settings.answer_prompt_version
        try:
            prompt = prompts.get("answer", version)
        except prompts.PromptError as exc:
            from app.core.errors import BadRequest

            raise BadRequest(str(exc)) from None
        retrieval = options.retrieval or RetrievalOptions(
            top_k=8,
            rerank=RerankOptions(enabled=self._providers.reranker is not None),
            context=ContextOptions(pool=12),
        )
        update: dict[str, Any] = {"context": retrieval.context.model_copy(update={"enabled": True})}
        if options.rewrite_follow_ups and has_history:
            update["transforms"] = retrieval.transforms.model_copy(update={"rewrite": True})
        retrieval = retrieval.model_copy(update=update)
        if retrieval.rerank.enabled and self._providers.reranker is None:
            from app.core.errors import BadRequest

            raise BadRequest(
                str(RerankUnavailable("Reranking was requested but none is configured"))
            )
        abstain_below = next(
            v
            for v in (
                options.abstain_below,
                policy.abstain_below,
                self._settings.answer_abstain_below,
            )
            if v is not None
        )
        return Prepared(
            collections=collections,
            retrieval=retrieval,
            pii=stricter_pii(policy.pii, options.pii),
            injection=stricter_injection(policy.injection, options.injection),
            abstain_below=abstain_below,
            prompt=prompt,
            options=options,
        )

    # -- the answer ---------------------------------------------------------------------------

    async def run(
        self,
        session: AsyncSession,
        workspace_id: uuid.UUID,
        question: str,
        prepared: Prepared,
        policy: WorkspacePolicy,
        *,
        history: Sequence[Turn] = (),
        summary_text: str = "",
        facts: Sequence[str] = (),
    ) -> AsyncIterator[Event]:
        try:
            async for event in self._run(
                session, workspace_id, question, prepared, policy, history, summary_text, facts
            ):
                yield event
        except ProviderError as exc:
            log.warning("answer failed: %s", exc)
            yield Event(
                "error",
                {"code": "provider_error", "message": str(exc), "provider": exc.provider,
                 "status": exc.status},
            )  # fmt: skip
        except RerankUnavailable as exc:
            yield Event("error", {"code": "bad_request", "message": str(exc)})

    async def _run(
        self, session, workspace_id, question, prepared, policy, history, summary_text, facts
    ) -> AsyncIterator[Event]:
        started = time.perf_counter()
        guardrails: dict[str, Any] = {
            "injection": [],
            "pii": {"mode": prepared.pii, "redacted": {}},
        }
        warnings: list[str] = []
        base = {
            "prompt": prepared.prompt.ref,
            "model": self._providers.llm.model,
            "warnings": warnings,
            "guardrails": guardrails,
        }

        # 1. the workspace's rules about what may be asked
        if term := violated_term(policy, question):
            guardrails["blocked_term"] = term
            yield Event(
                "done", self._refusal(base, policy.refusal_message, started, "blocked_question")
            )
            return

        redactor = Redactor() if prepared.pii != "off" else None
        redact_all = prepared.pii == "all" and redactor is not None
        asked = redactor.redact(question) if redact_all else question

        # 2. retrieval
        pipeline = RetrievalPipeline(session, self._providers, self._blobs)
        turns = list(history)
        if redact_all:
            turns = [Turn(role=t.role, content=redactor.redact(t.content)) for t in turns]
        result = await pipeline.run(
            asked,
            workspace_id,
            prepared.collections,
            prepared.retrieval,
            prepared.options.filters,
            turns,
        )
        retrieved_ms = (time.perf_counter() - started) * 1000
        warnings.extend(result.warnings)
        blocks = list(result.context.blocks) if result.context else []

        # 3. guardrails on the retrieved passages
        flagged: dict[str, str] = {}
        if prepared.injection != "off":
            for block in blocks:
                findings = screen(block.text)
                if findings:
                    reason = summary(findings)
                    action = "dropped" if prepared.injection == "drop" else "flagged"
                    guardrails["injection"].append(
                        {
                            "label": block.label,
                            "title": block.title,
                            "rules": [f.rule for f in findings],
                            "severity": max(f.severity for f in findings),
                            "action": action,
                        }
                    )
                    flagged[block.label] = reason
            if prepared.injection == "drop":
                blocks = [b for b in blocks if b.label not in flagged]
                flagged = {}
        texts = None
        if redactor is not None:  # model-facing text only; sources are returned as stored
            texts = {b.label: redactor.redact(b.text) for b in blocks}

        # 4. is there anything to answer from?
        before = conf.assess(asked, result.hits, blocks, abstain_below=prepared.abstain_below)
        # sources are shown as stored, except when the policy says personal data may not leave
        # the server at all
        sources = [
            source_dict(b, flagged.get(b.label), texts[b.label] if redact_all else None)
            for b in blocks
        ]
        yield Event(
            "retrieval",
            {
                "hits": [hit_dict(h) for h in result.hits],
                "sources": sources,
                "confidence": confidence_dict(before),
                "inferred_filters": [f.model_dump() for f in result.inferred],
                "guardrails": guardrails,
                "warnings": list(warnings),
            },
        )
        trace = result.trace.to_dict()
        if redact_all:
            for card in trace["chunks"].values():
                card["preview"] = redactor.redact(card["preview"])
        if not blocks or before.abstain:
            reason = "no_sources" if not blocks else "low_confidence"
            yield Event(
                "done",
                self._abstained(
                    base, reason, NOT_FOUND_MESSAGE, before, blocks, sources, trace, started,
                    retrieved_ms, policy,
                ),
            )  # fmt: skip
            return

        # 5. the prompt
        canary = "CANARY-" + secrets.token_hex(6)
        system = prepared.prompt.render(
            today=datetime.now(UTC).date().isoformat(),
            canary=canary,
            instructions=f"## Workspace rules\n{policy.instructions}"
            if policy.instructions
            else "",
            memory=self._facts(facts, redactor if redact_all else None),
            summary=f"## Conversation so far\n{summary_text}" if summary_text else "",
        )
        sources_text = render_sources(blocks, texts=texts, flagged=flagged)
        messages = [ChatMessage("system", system, cache=True)]
        messages += [ChatMessage(t.role, t.content) for t in turns]
        messages.append(ChatMessage("user", user_message(sources_text, asked)))

        # 6. generate, streaming; hold the text back when it must be checked before release
        buffered = bool(policy.blocked_terms) or redact_all
        gate, scanner = PrefixGate(), CitationScanner({b.label for b in blocks})
        by_label = {b.label: b for b in blocks}
        parts: list[str] = []
        usage: Usage | None = None
        first_token_ms: float | None = None
        generation_started = time.perf_counter()
        leaked = False
        async for chunk in self._providers.llm.stream(
            messages,
            temperature=prepared.options.temperature,
            max_tokens=prepared.options.max_tokens,
        ):
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.text:
                continue
            parts.append(chunk.text)
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            if canary in "".join(parts[-4:]) or canary in "".join(parts):
                leaked = True
                break  # stop reading: nothing more of this reply will be shown
            shown = gate.feed(chunk.text)
            if shown and not buffered:
                yield Event("token", {"text": shown})
            for label in scanner.feed(chunk.text):
                if not buffered and not gate.suppressed:
                    yield Event("citation", _stream_citation(by_label[label]))
        tail = gate.flush()
        if tail and not buffered and not leaked:
            yield Event("token", {"text": tail})
        generated_ms = (time.perf_counter() - generation_started) * 1000
        raw = "".join(parts).strip()

        # 7. check and finish the answer: each exit below is a different kind of ending
        if redactor is not None:
            guardrails["pii"]["redacted"] = redactor.found
        usage_data = self._usage(usage, started, retrieved_ms, generated_ms, first_token_ms)
        if leaked:
            guardrails["prompt_leak"] = True
            refusal = self._refusal(base, LEAK_MESSAGE, started, "prompt_leak", usage_data)
            yield Event("done", {**refusal, "sources": sources, "retrieval_trace": trace})
            return
        if not raw:
            warnings.append("The model returned an empty reply")
            yield Event(
                "done",
                self._abstained(
                    base, "empty_answer", NO_ANSWER_MESSAGE, before, blocks, sources, trace,
                    started, retrieved_ms, policy, usage_data,
                ),
            )  # fmt: skip
            return
        if raw.upper().startswith(NOT_FOUND):
            note = raw[len(NOT_FOUND) :].lstrip(" :.-\n").strip()
            yield Event(
                "done",
                self._abstained(
                    base, "model_not_found", f"{NOT_FOUND_MESSAGE} {note}".strip(), before,
                    blocks, sources, trace, started, retrieved_ms, policy, usage_data,
                ),
            )  # fmt: skip
            return

        grounding = ground(raw, blocks)
        answer = grounding.answer
        if redact_all:
            answer = redactor.redact(answer)
            guardrails["pii"]["redacted"] = redactor.found
        if (term := violated_term(policy, answer)) is not None:
            guardrails["blocked_answer_term"] = term
            refusal = self._refusal(
                base, policy.refusal_message, started, "blocked_answer", usage_data
            )
            yield Event("done", {**refusal, "sources": sources, "retrieval_trace": trace})
            return
        if buffered:  # released only now that the whole answer passed its checks
            for piece in split_answer(answer):
                yield Event("token", {"text": piece})
            for label in scanner.seen:
                yield Event("citation", _stream_citation(by_label[label]))
        if policy.disclaimer:
            answer = f"{answer}\n\n{policy.disclaimer}"
        finished = conf.finalize(before, grounding, abstain_below=prepared.abstain_below)
        yield Event(
            "done",
            {
                **base,
                "answer": answer,
                "abstained": False,
                "refused": False,
                "reason": None,
                "citations": [citation_dict(c) for c in grounding.citations],
                "grounding": grounding_dict(grounding, redactor.redact if redact_all else None),
                "confidence": confidence_dict(finished),
                "sources": sources,
                "related": [],
                "inferred_filters": [f.model_dump() for f in result.inferred],
                "usage": usage_data,
                "retrieval_trace": trace,
            },
        )

    # -- pieces -------------------------------------------------------------------------------

    def _facts(self, facts: Sequence[str], redactor: Redactor | None) -> str:
        shown = [f for f in facts if f.strip()]
        if not shown:
            return ""
        lines = [redactor.redact(f) if redactor else f for f in shown]
        return "## Known facts\nThese were set by the team; treat them as true.\n" + "\n".join(
            f"- {line}" for line in lines
        )

    def _usage(self, usage, started, retrieved_ms, generated_ms, first_token_ms) -> dict[str, Any]:
        u = usage or Usage()
        return {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "cached_tokens": u.cached_tokens,
            "latency_ms": {
                "retrieval": round(retrieved_ms, 1),
                "generation": round(generated_ms, 1),
                "first_token": None if first_token_ms is None else round(first_token_ms, 1),
                "total": round((time.perf_counter() - started) * 1000, 1),
            },
        }

    def _refusal(self, base, message, started, reason, usage=None) -> dict[str, Any]:
        return {
            **base,
            "answer": message,
            "abstained": False,
            "refused": True,
            "reason": reason,
            "citations": [],
            "grounding": None,
            "confidence": confidence_dict(conf.Confidence(0.0, "none", {}, [reason])),
            "sources": [],
            "related": [],
            "inferred_filters": [],
            "usage": usage or self._usage(None, started, 0.0, 0.0, None),
            "retrieval_trace": None,
        }

    def _abstained(
        self, base, reason, message, confidence, blocks, sources, trace, started, retrieved_ms,
        policy, usage=None,
    ) -> dict[str, Any]:  # fmt: skip
        related = [
            {
                "label": b.label,
                "title": b.title,
                "page": b.page,
                "document_id": str(b.document_id),
                "heading_path": b.heading_path,
            }
            for b in blocks[:RELATED_SOURCES]
        ]
        answer = message
        if policy.disclaimer:
            answer = f"{answer}\n\n{policy.disclaimer}"
        return {
            **base,
            "answer": answer,
            "abstained": True,
            "refused": False,
            "reason": reason,
            "citations": [],
            "grounding": None,
            "confidence": confidence_dict(confidence),
            "sources": sources,
            "related": related,
            "inferred_filters": [],
            "usage": usage or self._usage(None, started, retrieved_ms, 0.0, None),
            "retrieval_trace": trace,
        }


def _stream_citation(block: ContextBlock) -> dict[str, Any]:
    return {
        "label": block.label,
        "document_id": str(block.document_id),
        "title": block.title,
        "page": block.page,
        "heading_path": block.heading_path,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "passages": [_passage(p) for p in block.passages],
    }
