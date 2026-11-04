import random

import pytest
from chunk_helpers import check_chunks, overlap_chars

from app.ingest.chunking import ChunkingConfig, estimate_tokens
from app.ingest.chunking.fixed import FixedChunker
from app.ingest.chunking.recursive import RecursiveChunker, split_span
from app.ingest.files import Kind
from app.ingest.parsers import ParsedDocument, parse

SENT = "The system stores every document safely and indexes it for later search. "
CFG = ChunkingConfig(target_tokens=64, overlap_tokens=12, max_tokens=128)


def doc_of(text: str, **kw) -> ParsedDocument:
    return ParsedDocument(text=text, **kw)


def prose(sentences: int, per_paragraph: int = 5) -> str:
    paragraphs = []
    for p in range(0, sentences, per_paragraph):
        n = min(per_paragraph, sentences - p)
        paragraphs.append(
            " ".join(f"Sentence {p + i} talks about topic {p // per_paragraph}." for i in range(n))
        )
    return "\n\n".join(paragraphs)


# --- fixed ----------------------------------------------------------------------------------


async def test_fixed_windows_have_the_target_size_and_overlap():
    parsed = doc_of(SENT * 60)
    drafts = await FixedChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(drafts) > 5
    assert all(d.token_count <= CFG.target_tokens + 2 for d in drafts)
    assert all(d.token_count >= CFG.target_tokens * 0.7 for d in drafts[:-1])
    assert all(o > 0 for o in overlap_chars(drafts))


async def test_fixed_never_splits_a_word():
    text = " ".join(f"word{i}" for i in range(500))
    parsed = doc_of(text)
    drafts = await FixedChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    for d in drafts:
        for token in d.text.split():
            assert token in text.split()  # every token is a whole original word


async def test_fixed_without_overlap_tiles_the_text():
    cfg = ChunkingConfig(target_tokens=64, overlap_tokens=0, max_tokens=128)
    parsed = doc_of(SENT * 40)
    drafts = await FixedChunker().chunk(parsed, cfg)
    check_chunks(parsed, drafts, cfg)
    assert all(o == 0 for o in overlap_chars(drafts))


async def test_fixed_short_and_empty_documents():
    assert await FixedChunker().chunk(doc_of(""), CFG) == []
    drafts = await FixedChunker().chunk(doc_of("Just a line."), CFG)
    assert [d.text for d in drafts] == ["Just a line."]


async def test_fixed_handles_text_without_any_whitespace():
    parsed = doc_of("x" * 5000)
    drafts = await FixedChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(drafts) > 3


async def test_fixed_sizes_cjk_by_character_density():
    parsed = doc_of("日本語のテキストを安全に保存して検索できるようにします。" * 80)
    drafts = await FixedChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert all(d.token_count <= CFG.max_tokens for d in drafts)


async def test_fixed_is_deterministic():
    parsed = doc_of(SENT * 30)
    a = await FixedChunker().chunk(parsed, CFG)
    b = await FixedChunker().chunk(parsed, CFG)
    assert [(d.char_start, d.char_end) for d in a] == [(d.char_start, d.char_end) for d in b]


# --- recursive ------------------------------------------------------------------------------


async def test_recursive_one_chunk_when_everything_fits():
    parsed = doc_of("A short document.\n\nWith two paragraphs.")
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    assert [d.text for d in drafts] == [parsed.text]


async def test_recursive_ends_chunks_at_natural_boundaries():
    parsed = doc_of(prose(60))
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(drafts) > 4
    for d in drafts:
        assert d.char_end == len(parsed.text) or parsed.text[d.char_end].isspace()
        assert d.text.endswith(".")  # sentence or paragraph end, never mid-sentence
        assert d.text[0].isupper()


async def test_recursive_splits_an_oversized_paragraph_by_sentence():
    parsed = doc_of(SENT * 50)  # one 600-token paragraph
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert all(d.text.endswith(".") for d in drafts)
    assert all(d.token_count <= CFG.target_tokens for d in drafts)


async def test_recursive_overlap_repeats_trailing_sentences():
    cfg = ChunkingConfig(target_tokens=64, overlap_tokens=24, max_tokens=128)
    parsed = doc_of(SENT * 50)
    drafts = await RecursiveChunker().chunk(parsed, cfg)
    check_chunks(parsed, drafts, cfg)
    overlaps = overlap_chars(drafts)
    assert sum(1 for o in overlaps if o > 0) >= len(overlaps) * 0.8
    for a, b in zip(drafts, drafts[1:], strict=False):
        if b.char_start < a.char_end:  # the shared text is whole sentences
            assert parsed.text[b.char_start - 1].isspace()


async def test_recursive_does_not_overlap_when_disabled():
    cfg = ChunkingConfig(target_tokens=64, overlap_tokens=0, max_tokens=128)
    parsed = doc_of(SENT * 50)
    drafts = await RecursiveChunker().chunk(parsed, cfg)
    assert all(o == 0 for o in overlap_chars(drafts))


def sectioned() -> ParsedDocument:
    from app.ingest.parsers import Heading

    parts, headings, offset = [], [], 0
    for level, title, body in [
        (1, "Guide", ""),
        (2, "Setup", prose(8)),
        (2, "Usage", prose(8)),
        (3, "Tiny", "One short line."),
        (2, "Reference", prose(8)),
    ]:
        block = title + ("\n\n" + body if body else "")
        headings.append(Heading(level, title, offset))
        parts.append(block)
        offset += len(block) + 2
    return doc_of("\n\n".join(parts), headings=headings)


async def test_recursive_keeps_headings_with_their_text():
    parsed = sectioned()
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    heading_texts = {h.text for h in parsed.headings}
    for d in drafts:
        # no chunk ends on a bare heading, and no chunk is just a heading
        assert d.text.split("\n")[-1].strip() not in heading_texts
        assert d.text not in heading_texts


async def test_recursive_starts_new_sections_in_new_chunks():
    parsed = sectioned()
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    for h in parsed.headings:
        # Only the big section after another big one: "Setup" follows an empty "Guide" and
        # "Reference" follows the tiny "Tiny", and tiny sections are merged forward on purpose.
        if h.text == "Usage":
            owner = [d for d in drafts if d.char_start <= h.char_start < d.char_end]
            assert owner and owner[0].char_start == h.char_start, h.text


async def test_recursive_merges_tiny_sections_instead_of_isolating_them():
    parsed = sectioned()
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    tiny = next(h for h in parsed.headings if h.text == "Tiny")
    owner = next(d for d in drafts if d.char_start <= tiny.char_start < d.char_end)
    assert owner.token_count > estimate_tokens("Tiny\n\nOne short line.")


async def test_recursive_heading_path_follows_the_outline():
    parsed = sectioned()
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    usage = next(d for d in drafts if "Usage" in d.heading_path[-1:])
    assert usage.heading_path == ["Guide", "Usage"]


async def test_recursive_hard_splits_an_unbreakable_token():
    parsed = doc_of("start " + "A" * 6000 + " end")
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert max(d.token_count for d in drafts) <= CFG.target_tokens


async def test_recursive_keeps_table_rows_and_list_items_whole():
    rows = "\n".join(f"name: user{i} | plan: pro | seats: {i}" for i in range(80))
    parsed = doc_of(rows)
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    for d in drafts:
        for line in d.text.split("\n"):
            assert line.startswith("name: user") and line.split("seats: ")[1].isdigit()


def test_split_span_prefers_the_coarsest_separator():
    text = "alpha beta.\n\ngamma delta. epsilon zeta.\n\neta theta."
    assert [text[a:b] for a, b in split_span(text, 0, len(text), 5)] == [
        "alpha beta.",
        "gamma delta.",
        "epsilon zeta.",
        "eta theta.",
    ]
    assert [text[a:b] for a, b in split_span(text, 0, len(text), 100)] == [text]


async def test_recursive_on_a_real_parsed_pdf():
    import pdfgen

    parsed = parse(pdfgen.report_pdf(), Kind.PDF)
    drafts = await RecursiveChunker().chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert {d.page for d in drafts} == {1, 2, 3}
    assert any(d.heading_path == ["Annual Report", "Financial results"] for d in drafts)


@pytest.mark.parametrize("chunker", [FixedChunker, RecursiveChunker])
async def test_random_documents_keep_every_invariant(chunker):
    rng = random.Random(f"fuzz-{chunker.__name__}")
    vocab = [
        "alpha",
        "beta.",
        "Gamma",
        "delta!",
        "x" * 40,
        "日本語。",
        "\n",
        "\n\n",
        "  ",
        "3.5",
        "Dr.",
    ]
    for _ in range(150):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randrange(1, 300))).strip() or "x"
        parsed = doc_of(text)
        check_chunks(parsed, await chunker().chunk(parsed, CFG), CFG)
