import random

import pytest
from chunk_helpers import check_chunks

from app.ingest.chunking import ChunkingConfig
from app.ingest.chunking.semantic import SemanticChunker, cosine_distance, percentile
from app.ingest.parsers import Heading, ParsedDocument
from app.providers.errors import ProviderError
from app.providers.fake import FakeEmbedder

CATS = [
    "Cats sleep most of the day and hunt at night.",
    "A cat grooms its fur with a rough tongue.",
    "Kittens learn to hunt by playing with toys.",
    "Many cats purr when they feel relaxed and safe.",
    "Cats use their whiskers to judge narrow gaps.",
    "A hungry cat will meow loudly near its bowl.",
]
TAXES = [
    "Income tax returns are due in April every year.",
    "Tax deductions reduce the income you owe tax on.",
    "An accountant can file your tax return for a fee.",
    "Late tax payments attract interest and penalties.",
    "Corporate tax rates differ between countries.",
    "Sales tax is added to the price of most goods.",
]
CFG = ChunkingConfig(target_tokens=64, overlap_tokens=0, max_tokens=160, breakpoint_percentile=85)


def doc_of(sentences: list[str], **kw) -> ParsedDocument:
    return ParsedDocument(text=" ".join(sentences), **kw)


def test_percentile_interpolates():
    assert percentile([], 90) == 0.0
    assert percentile([5], 90) == 5
    assert percentile([0, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([4, 1, 3, 2], 50) == 2.5


def test_cosine_distance():
    assert cosine_distance([1, 0], [1, 0]) == pytest.approx(0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1)
    assert cosine_distance([0, 0], [1, 0]) == 1.0  # a zero vector is maximally distant


async def test_cuts_at_the_topic_change():
    parsed = doc_of(CATS + TAXES)
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(drafts) >= 2
    cat_words = ("cat", "kitten", "whisker")
    for d in drafts:
        has_cats = any(w in d.text.lower() for w in cat_words)
        has_tax = "tax" in d.text.lower()
        assert not (has_cats and has_tax), d.text  # no chunk mixes the two topics


async def test_three_topics_give_three_groups():
    parsed = doc_of(CATS + TAXES + CATS)
    cfg = ChunkingConfig(
        target_tokens=64, overlap_tokens=0, max_tokens=200, breakpoint_percentile=80
    )
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, cfg)
    check_chunks(parsed, drafts, cfg)
    assert len(drafts) >= 3


async def test_a_short_document_is_a_single_chunk_without_embedding():
    class Exploding(FakeEmbedder):
        async def embed(self, texts):
            raise AssertionError("must not be called for a document that fits in one chunk")

    parsed = doc_of(["One short sentence.", "And another."])
    drafts = await SemanticChunker(Exploding()).chunk(parsed, CFG)
    assert [d.text for d in drafts] == [parsed.text]


async def test_empty_document():
    assert await SemanticChunker(FakeEmbedder()).chunk(ParsedDocument(text=""), CFG) == []


async def test_uniform_text_is_only_cut_by_size():
    parsed = doc_of(["The same sentence repeats again and again."] * 60)
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert all(d.token_count <= CFG.max_tokens for d in drafts)
    assert len(drafts) > 1


async def test_chunks_are_not_smaller_than_the_minimum_except_possibly_the_last():
    parsed = doc_of(CATS + TAXES + CATS + TAXES)
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
    assert all(d.token_count >= CFG.min_tokens for d in drafts[:-1])


async def test_new_sections_start_new_chunks_when_the_chunk_is_big_enough():
    cats = " ".join(CATS)
    taxes = " ".join(TAXES)
    text = f"Pets\n\n{cats}\n\nMoney\n\n{taxes}"
    parsed = ParsedDocument(
        text=text,
        headings=[Heading(1, "Pets", 0), Heading(1, "Money", text.index("Money"))],
    )
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    money = text.index("Money")
    assert any(d.char_start == money for d in drafts)
    assert any(d.heading_path == ["Money"] for d in drafts)


async def test_a_sentence_longer_than_a_chunk_is_split():
    parsed = ParsedDocument(text="start " + "word " * 900 + "end.")
    drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
    check_chunks(parsed, drafts, CFG)
    assert len(drafts) > 2


async def test_embeds_in_batches_with_neighbour_context():
    seen: list[list[str]] = []

    class Spy(FakeEmbedder):
        async def embed(self, texts):
            seen.append(list(texts))
            return await super().embed(texts)

    parsed = doc_of(CATS * 20)  # 120 sentences -> two batches of <= 64
    await SemanticChunker(Spy()).chunk(parsed, CFG)
    assert [len(b) for b in seen] == [64, 56]
    first = seen[0][1]  # sentence 1 is embedded together with sentences 0 and 2
    assert CATS[0] in first and CATS[1] in first and CATS[2] in first


async def test_provider_errors_propagate_so_the_job_can_retry():
    class Down(FakeEmbedder):
        async def embed(self, texts):
            raise ProviderError("ollama", "connection refused")

    with pytest.raises(ProviderError):
        await SemanticChunker(Down()).chunk(doc_of(CATS + TAXES), CFG)


async def test_a_misbehaving_embedder_is_detected():
    class Short(FakeEmbedder):
        async def embed(self, texts):
            return (await super().embed(texts))[:-1]

    with pytest.raises(ValueError, match="different number"):
        await SemanticChunker(Short()).chunk(doc_of(CATS + TAXES), CFG)


async def test_random_documents_keep_every_invariant():
    rng = random.Random("semantic")
    pool = CATS + TAXES + ["x" * 30, "Short.", "Dr. Who arrived.", "日本語のテキストです。"]
    for _ in range(40):
        parsed = doc_of([rng.choice(pool) for _ in range(rng.randrange(1, 80))])
        drafts = await SemanticChunker(FakeEmbedder()).chunk(parsed, CFG)
        check_chunks(parsed, drafts, CFG)
