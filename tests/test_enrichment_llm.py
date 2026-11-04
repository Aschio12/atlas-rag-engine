import asyncio
import json

import pytest

from app.ingest.enrichment.config import EnrichmentConfig
from app.ingest.enrichment.llm import (
    build_messages,
    enrich_with_llm,
    parse_reply,
)
from app.providers.base import LLM, LLMResponse, Usage
from app.providers.errors import ProviderError

BOTH = EnrichmentConfig(summaries=True, questions=True, max_questions=3)


def reply(summary="Refunds are allowed within 14 days.", questions=None):
    return json.dumps(
        {
            "summary": summary,
            "questions": questions
            or ["How long do I have to request a refund?", "Can I get my money back?"],
        }
    )


class Scripted(LLM):
    name, model = "scripted", "scripted"

    def __init__(self, fn):
        self.fn, self.calls, self.running, self.peak = fn, [], 0, 0

    async def complete(self, messages, *, temperature=0.2, max_tokens=1024):
        self.calls.append(messages)
        n = len(self.calls)  # this call's number, fixed before any concurrent call can change it
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return LLMResponse(self.fn(n, messages), "m", Usage(50, 10))


# --- parsing --------------------------------------------------------------------------------


def test_parses_a_clean_reply():
    out = parse_reply(reply(), BOTH)
    assert out.summary == "Refunds are allowed within 14 days."
    assert out.questions == ["How long do I have to request a refund?", "Can I get my money back?"]


def test_tolerates_prose_and_code_fences_around_the_json():
    raw = f"Sure! Here you go:\n```json\n{reply()}\n```\nHope that helps."
    assert parse_reply(raw, BOTH).summary.startswith("Refunds")


@pytest.mark.parametrize(
    "raw",
    ["", "no json here", "{broken json", "[1, 2, 3]", '"just a string"', "{}", '{"summary": ""}'],
)
def test_unusable_replies_are_rejected(raw):
    assert parse_reply(raw, BOTH) is None


def test_only_requested_fields_are_kept():
    only_summary = EnrichmentConfig(summaries=True, questions=False)
    out = parse_reply(reply(), only_summary)
    assert out.summary and out.questions == []
    only_q = EnrichmentConfig(summaries=False, questions=True)
    out = parse_reply(reply(), only_q)
    assert out.summary is None and len(out.questions) == 2


def test_junk_questions_are_filtered_and_capped():
    qs = [
        "How long is the refund window?",
        "how long is the refund window",  # duplicate (case/punctuation)
        "Statement, not a question.",
        "Too short?",
        "x" * 300 + "?",
        42,
        "What does the policy cover?",
        "Who pays for return shipping?",
        "Is there a fee for refunds?",
    ]
    out = parse_reply(
        json.dumps({"questions": qs}), EnrichmentConfig(questions=True, max_questions=3)
    )
    assert out.questions == [
        "How long is the refund window?",
        "What does the policy cover?",
        "Who pays for return shipping?",
    ]


def test_long_summaries_are_trimmed_and_whitespace_normalised():
    out = parse_reply(json.dumps({"summary": "word  " * 200}), EnrichmentConfig(summaries=True))
    assert len(out.summary) <= 300 and "  " not in out.summary


def test_fullwidth_question_marks_are_accepted():
    out = parse_reply(json.dumps({"questions": ["返金はいつまで可能ですか？"]}), BOTH)
    assert out.questions == ["返金はいつまで可能ですか？"]


def test_prompt_asks_only_for_the_enabled_fields_and_includes_context():
    msgs = build_messages(
        ["Handbook", "Refunds"],
        "Refunds within 14 days.",
        EnrichmentConfig(questions=True, max_questions=2),
    )
    user = msgs[1].content
    assert "Handbook > Refunds" in user and "Refunds within 14 days." in user
    assert '"questions"' in user and "up to 2" in user and '"summary"' not in user
    assert msgs[0].role == "system"


# --- orchestration --------------------------------------------------------------------------


async def test_results_are_returned_in_input_order_with_usage():
    llm = Scripted(
        lambda n, m: reply(summary=f"Summary {m[1].content.split('<passage>')[1].split()[0]}")
    )
    items = [([], f"chunk{i} text") for i in range(6)]
    results, report = await enrich_with_llm(llm, items, BOTH)
    assert [r.summary for r in results] == [f"Summary chunk{i}" for i in range(6)]
    assert (report.calls, report.failed, report.invalid, report.skipped) == (6, 0, 0, 0)
    assert (report.prompt_tokens, report.completion_tokens) == (300, 60)


async def test_concurrency_is_bounded():
    llm = Scripted(lambda n, m: reply())
    cfg = EnrichmentConfig(summaries=True, concurrency=3)
    await enrich_with_llm(llm, [([], f"c{i}") for i in range(12)], cfg)
    assert 1 < llm.peak <= 3


async def test_the_cap_skips_the_rest():
    llm = Scripted(lambda n, m: reply())
    cfg = EnrichmentConfig(summaries=True, max_chunks=2)
    results, report = await enrich_with_llm(llm, [([], f"c{i}") for i in range(5)], cfg)
    assert len(llm.calls) == 2 and report.skipped == 3
    assert [r is not None for r in results] == [True, True, False, False, False]


async def test_bad_replies_are_counted_not_stored():
    llm = Scripted(lambda n, m: "I cannot do that" if n % 2 else reply())
    results, report = await enrich_with_llm(llm, [([], f"c{i}") for i in range(4)], BOTH)
    assert report.invalid == 2
    assert sum(r is not None for r in results) == 2


async def test_isolated_failures_only_cost_those_chunks():
    def fn(n, m):
        if n == 2:
            raise ProviderError("anthropic", "overloaded", 529)
        return reply()

    llm = Scripted(fn)
    results, report = await enrich_with_llm(llm, [([], f"c{i}") for i in range(5)], BOTH)
    assert report.failed == 1
    assert sum(r is None for r in results) == 1


async def test_a_dead_provider_raises_so_the_job_retries():
    def fn(n, m):
        raise ProviderError("ollama", "connection refused")

    llm = Scripted(fn)
    cfg = EnrichmentConfig(summaries=True, concurrency=1)
    with pytest.raises(ProviderError):
        await enrich_with_llm(llm, [([], f"c{i}") for i in range(20)], cfg)
    assert len(llm.calls) == 3  # gave up fast


async def test_nothing_happens_when_llm_enrichers_are_off_or_there_are_no_items():
    llm = Scripted(lambda n, m: reply())
    results, report = await enrich_with_llm(llm, [([], "x")], EnrichmentConfig())
    assert results == [None] and report.calls == 0 and llm.calls == []
    assert await enrich_with_llm(llm, [], BOTH) == ([], report)
