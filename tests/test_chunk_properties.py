"""Every strategy x every document format, plus scale and hostile input."""

import random
import time

import pytest
from chunk_helpers import check_chunks
from test_ingest_pipeline import CASES
from test_parser_properties import KINDS

from app.db.models import ChunkingStrategy
from app.ingest.chunking import ChunkingConfig, build_chunker
from app.ingest.parsers import Heading, ParsedDocument, parse
from app.providers.fake import FakeLLM
from app.providers.registry import Providers

CFG = ChunkingConfig(
    target_tokens=64, overlap_tokens=8, max_tokens=128, parent_tokens=256, child_tokens=64
)


@pytest.fixture(scope="module")
def providers():
    from app.core.config import Settings

    p = Providers(
        Settings(_env_file=None, llm_provider="fake", embed_provider="fake", vector_store="memory")
    )
    p._llm = FakeLLM(reply="Situating context.")
    return p


def split_levels(drafts):
    return (
        [d for d in drafts if d.kind != "parent"],
        [d for d in drafts if d.kind == "parent"],
    )


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
@pytest.mark.parametrize("fmt", list(CASES))
async def test_every_strategy_on_every_format_keeps_the_invariants(strategy, fmt, providers):
    _, data, *_ = CASES[fmt]
    parsed = parse(data, KINDS[fmt])
    drafts = await build_chunker(strategy, providers).chunk(parsed, CFG)
    assert drafts
    searchable, parents = split_levels(drafts)
    check_chunks(parsed, searchable, CFG)
    if parents:
        check_chunks(parsed, parents, CFG, max_tokens=int(CFG.parent_tokens * 1.25))
        for d in searchable:
            parent = drafts[d.parent]
            assert parent.char_start <= d.char_start and d.char_end <= parent.char_end


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
async def test_chunking_is_deterministic(strategy, providers):
    parsed = parse(CASES["md"][1], KINDS["md"])
    runs = [await build_chunker(strategy, providers).chunk(parsed, CFG) for _ in range(2)]
    assert [(d.char_start, d.char_end, d.kind) for d in runs[0]] == [
        (d.char_start, d.char_end, d.kind) for d in runs[1]
    ]


@pytest.mark.parametrize("strategy", [s for s in ChunkingStrategy if s.value != "contextual"])
async def test_hostile_text_never_breaks_a_chunker(strategy, providers):
    rng = random.Random(f"hostile-{strategy}")
    vocab = ["a", "Word.", "…", "日本語。", "\n", "\n\n\n", "x" * 90, "# H", "|", "Dr.", "\t", "1."]
    for _ in range(40):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randrange(1, 200))).strip() or "x"
        parsed = ParsedDocument(text=text)
        drafts = await build_chunker(strategy, providers).chunk(parsed, CFG)
        searchable, parents = split_levels(drafts)
        check_chunks(parsed, searchable, CFG)
        if parents:
            check_chunks(parsed, parents, CFG, max_tokens=int(CFG.parent_tokens * 1.25))


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "parent_child"])
async def test_a_megabyte_of_text_chunks_quickly(strategy, providers):
    """Guards against accidental quadratic behaviour: ~1 MB with 240 headings."""
    paras = [
        f"Paragraph {i} explains the policy for team {i}. It has two sentences."
        for i in range(15000)
    ]
    text, headings, offset = "\n\n".join(paras), [], 0
    for i, p in enumerate(paras):
        if i % 60 == 0:
            headings.append(Heading(2, f"Section {i}", offset))
        offset += len(p) + 2
    parsed = ParsedDocument(text=text, headings=headings)
    started = time.perf_counter()
    drafts = await build_chunker(ChunkingStrategy(strategy), providers).chunk(
        parsed, ChunkingConfig()
    )
    assert time.perf_counter() - started < 15
    searchable, _ = split_levels(drafts)
    check_chunks(parsed, searchable, ChunkingConfig(), max_tokens=ChunkingConfig().max_tokens)
