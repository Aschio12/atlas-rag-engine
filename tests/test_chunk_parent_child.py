from chunk_helpers import check_chunks

from app.ingest.chunking import ChunkingConfig
from app.ingest.chunking.parent_child import ParentChildChunker
from app.ingest.parsers import Heading, ParsedDocument

CFG = ChunkingConfig(
    target_tokens=64, overlap_tokens=8, max_tokens=128, parent_tokens=256, child_tokens=64
)


def prose(n: int) -> str:
    paragraphs = []
    for p in range(0, n, 4):
        paragraphs.append(
            " ".join(f"Sentence {p + i} explains part {p // 4} in some detail." for i in range(4))
        )
    return "\n\n".join(paragraphs)


def split(drafts):
    return [d for d in drafts if d.kind == "parent"], [d for d in drafts if d.kind == "child"]


async def test_children_belong_to_parents_that_contain_them():
    parsed = ParsedDocument(text=prose(80))
    drafts = await ParentChildChunker().chunk(parsed, CFG)
    parents, children = split(drafts)
    assert len(parents) >= 3 and len(children) > len(parents) * 2
    for child in children:
        parent = drafts[child.parent]
        assert parent.kind == "parent"
        assert parent.char_start <= child.char_start and child.char_end <= parent.char_end


async def test_every_parent_has_children_and_no_orphans():
    parsed = ParsedDocument(text=prose(80))
    drafts = await ParentChildChunker().chunk(parsed, CFG)
    parents = {i for i, d in enumerate(drafts) if d.kind == "parent"}
    used = {d.parent for d in drafts if d.kind == "child"}
    assert parents == used
    assert all(d.parent is None for d in drafts if d.kind == "parent")


async def test_parents_are_big_children_are_small():
    parsed = ParsedDocument(text=prose(80))
    parents, children = split(await ParentChildChunker().chunk(parsed, CFG))
    assert max(c.token_count for c in children) <= CFG.max_tokens
    assert max(p.token_count for p in parents) <= int(CFG.parent_tokens * 1.25)
    assert sum(p.token_count for p in parents) / len(parents) > 2 * (
        sum(c.token_count for c in children) / len(children)
    )


async def test_parents_are_not_indexable_children_are():
    parsed = ParsedDocument(text=prose(40))
    drafts = await ParentChildChunker().chunk(parsed, CFG)
    assert all(not d.indexable for d in drafts if d.kind == "parent")
    assert all(d.indexable for d in drafts if d.kind == "child")


async def test_children_and_parents_each_satisfy_the_chunk_invariants():
    parsed = ParsedDocument(text=prose(80))
    parents, children = split(await ParentChildChunker().chunk(parsed, CFG))
    check_chunks(parsed, children, CFG)
    check_chunks(parsed, parents, CFG, max_tokens=int(CFG.parent_tokens * 1.25))


async def test_parents_tile_the_document_without_overlap():
    parsed = ParsedDocument(text=prose(80))
    parents, _ = split(await ParentChildChunker().chunk(parsed, CFG))
    for a, b in zip(parents, parents[1:], strict=False):
        assert a.char_end <= b.char_start


async def test_a_small_document_has_one_parent_and_one_child():
    parsed = ParsedDocument(text="A tiny document with one idea.")
    drafts = await ParentChildChunker().chunk(parsed, CFG)
    assert [d.kind for d in drafts] == ["parent", "child"]
    assert drafts[0].text == drafts[1].text == parsed.text
    assert drafts[1].parent == 0


async def test_empty_document():
    assert await ParentChildChunker().chunk(ParsedDocument(text=""), CFG) == []


async def test_headings_and_pages_carry_to_both_levels():
    text = "Intro\n\n" + prose(40) + "\n\nDetails\n\n" + prose(40)
    parsed = ParsedDocument(
        text=text,
        headings=[Heading(1, "Intro", 0), Heading(1, "Details", text.index("Details"))],
    )
    drafts = await ParentChildChunker().chunk(parsed, CFG)
    late = [d for d in drafts if d.char_start >= text.index("Details")]
    assert late and all(d.heading_path == ["Details"] for d in late)
