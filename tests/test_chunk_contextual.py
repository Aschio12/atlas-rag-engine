import pytest
from chunk_helpers import check_chunks

from app.ingest.chunking import ChunkingConfig
from app.ingest.chunking.contextual import (
    WINDOW_CHARS,
    ContextualChunker,
    clean_context,
)
from app.ingest.parsers import ParsedDocument
from app.providers.base import LLM, LLMResponse, Usage
from app.providers.errors import ProviderError
from app.providers.fake import FakeLLM

CFG = ChunkingConfig(target_tokens=64, overlap_tokens=0, max_tokens=128)


def doc(n=30, title=None) -> ParsedDocument:
    paragraphs = [
        f"Paragraph {i} describes the refund rules for plan {i} in detail." for i in range(n)
    ]
    return ParsedDocument(text="\n\n".join(paragraphs), title=title)


class ScriptedLLM(LLM):
    name = "scripted"
    model = "scripted"

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    async def complete(self, messages, *, temperature=0.2, max_tokens=1024):
        self.calls.append((messages, temperature, max_tokens))
        return self.fn(len(self.calls), messages)


def ok(text="Part of the refund policy.", cached=0):
    return LLMResponse(text, "m", Usage(100, 8, cached))


async def test_every_chunk_gets_a_context_and_text_is_untouched():
    parsed = doc()
    llm = ScriptedLLM(lambda n, m: ok(f"Context number {n}."))
    drafts = await ContextualChunker(llm).chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(llm.calls) == len(drafts) > 3
    assert [d.context for d in drafts] == [f"Context number {i + 1}." for i in range(len(drafts))]
    for d in drafts:
        assert parsed.text[d.char_start : d.char_end] == d.text  # context never alters the text
        assert d.embed_text == f"{d.context}\n\n{d.text}"


async def test_the_document_is_a_cached_prefix_shared_by_all_calls():
    parsed = doc(title="Refund Handbook")
    llm = ScriptedLLM(lambda n, m: ok())
    drafts = await ContextualChunker(llm).chunk(parsed, CFG)
    first = llm.calls[0][0]
    assert first[0].role == "system" and first[0].cache is True
    assert "Title: Refund Handbook" in first[0].content
    assert parsed.text[:200] in first[0].content
    assert first[1].role == "user" and drafts[0].text in first[1].content and not first[1].cache
    # identical prefix on every call keeps the provider cache warm
    assert len({c[0][0].content for c in llm.calls}) == 1
    assert all(c[1] == 0.0 for c in llm.calls)  # deterministic


async def test_long_documents_use_one_cached_prefix_per_window():
    para = "A long paragraph about refunds and returns. " * 20
    parsed = ParsedDocument(text="\n\n".join([para] * 120))  # ~100k characters
    assert len(parsed.text) > 2 * WINDOW_CHARS
    llm = ScriptedLLM(lambda n, m: ok())
    cfg = ChunkingConfig(
        target_tokens=400, overlap_tokens=0, max_tokens=800, context_max_chunks=200
    )
    drafts = await ContextualChunker(llm).chunk(parsed, cfg)
    prefixes = {c[0][0].content for c in llm.calls}
    assert len(prefixes) == 3
    assert all(len(p) <= WINDOW_CHARS + 8_000 + 400 for p in prefixes)
    for d, call in zip(drafts, llm.calls, strict=True):  # each chunk is inside its own prefix
        assert d.text in call[0][0].content


async def test_usage_including_cache_reads_is_reported():
    llm = ScriptedLLM(lambda n, m: ok(cached=0 if n == 1 else 90))
    chunker = ContextualChunker(llm)
    drafts = await chunker.chunk(doc(), CFG)
    r = chunker.report
    assert r.calls == len(drafts)
    assert (r.failed, r.skipped) == (0, 0)
    assert r.prompt_tokens == 100 * len(drafts)
    assert r.completion_tokens == 8 * len(drafts)
    assert r.cached_tokens == 90 * (len(drafts) - 1)


async def test_the_cost_cap_limits_llm_calls():
    cfg = ChunkingConfig(target_tokens=64, overlap_tokens=0, max_tokens=128, context_max_chunks=3)
    llm = ScriptedLLM(lambda n, m: ok())
    chunker = ContextualChunker(llm)
    drafts = await chunker.chunk(doc(), cfg)
    assert len(llm.calls) == 3
    assert [d.context is not None for d in drafts[:4]] == [True, True, True, False]
    assert chunker.report.skipped == len(drafts) - 3


async def test_an_occasional_failure_only_costs_that_chunk_its_context():
    def fn(n, messages):
        if n == 2:
            raise ProviderError("anthropic", "overloaded", 529)
        return ok()

    llm = ScriptedLLM(fn)
    chunker = ContextualChunker(llm)
    drafts = await chunker.chunk(doc(), CFG)
    assert drafts[1].context is None
    assert all(d.context for i, d in enumerate(drafts) if i != 1)
    assert chunker.report.failed == 1


async def test_a_provider_that_is_down_aborts_so_the_job_can_retry():
    def fn(n, messages):
        raise ProviderError("ollama", "connection refused")

    llm = ScriptedLLM(fn)
    with pytest.raises(ProviderError):
        await ContextualChunker(llm).chunk(doc(), CFG)
    assert len(llm.calls) == 3  # gave up quickly instead of failing every chunk


async def test_blank_replies_leave_no_context():
    llm = ScriptedLLM(lambda n, m: ok("   "))
    drafts = await ContextualChunker(llm).chunk(doc(5), CFG)
    assert all(d.context is None for d in drafts)


async def test_works_with_the_fake_llm_and_empty_documents():
    drafts = await ContextualChunker(FakeLLM(reply="Situating sentence.")).chunk(doc(), CFG)
    assert all(d.context == "Situating sentence." for d in drafts)
    assert await ContextualChunker(FakeLLM()).chunk(ParsedDocument(text=""), CFG) == []


def test_clean_context():
    assert clean_context('  "Context: From the 2025 report.\n\nSection 2."  ') == (
        "From the 2025 report. Section 2."
    )
    assert clean_context("") is None
    long = clean_context("word " * 200)
    assert len(long) <= 400 and long.endswith("…")
