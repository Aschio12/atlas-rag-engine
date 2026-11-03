from app.ingest.parsers.base import Heading, PageSpan, ParsedDocument, TextBuilder, normalize


def test_normalize_unicode_newlines_and_whitespace():
    raw = "Café\r\nline  with\t\tgaps \r\r\n\n\n\n\nend​\x00\x07"
    assert normalize(raw) == "Café\nline with gaps\n\nend"


def test_normalize_nbsp_and_zero_width_and_bom():
    assert normalize("a b﻿​c") == "a bc"


def test_normalize_collapses_blank_runs_but_keeps_one():
    assert normalize("a\n\n\n\n\nb") == "a\n\nb"


def test_normalize_keep_indent_preserves_code_layout():
    code = "def f():\n\treturn 1\n    x"
    assert normalize(code, keep_indent=True) == "def f():\n    return 1\n    x"


def test_normalize_is_idempotent():
    messy = "A  B\r\n\r\n\r\n\tC  \n"
    once = normalize(messy)
    assert normalize(once) == once


def test_builder_records_heading_offsets_exactly():
    b = TextBuilder()
    b.add("Intro", heading_level=1)
    b.add("Some body text.")
    b.add("Details", heading_level=2)
    b.add("More.")
    text, _, headings = b.build()
    assert text == "Intro\n\nSome body text.\n\nDetails\n\nMore."
    for h in headings:
        assert text[h.char_start : h.char_start + len(h.text)] == h.text
    assert [(h.level, h.text) for h in headings] == [(1, "Intro"), (2, "Details")]


def test_builder_skips_empty_blocks_and_clamps_levels():
    b = TextBuilder()
    assert b.add("   \n ") is None
    b.add("Deep", heading_level=9)
    b.add("Top", heading_level=0)
    _, _, headings = b.build()
    assert [h.level for h in headings] == [6, 1]


def test_builder_separator_is_configurable():
    b = TextBuilder()
    b.add("row1")
    b.add("row2", sep="\n")
    assert b.build()[0] == "row1\nrow2"


def test_builder_pages_partition_the_text_without_separators():
    b = TextBuilder()
    b.start_page(1)
    b.add("Page one text.")
    b.add("More one.")
    b.start_page(2)
    b.add("Page two.")
    b.start_page(3)  # an empty page
    b.start_page(4)
    b.add("Page four.")
    text, pages, _ = b.build()
    assert [p.page for p in pages] == [1, 2, 3, 4]
    assert text[pages[0].char_start : pages[0].char_end] == "Page one text.\n\nMore one."
    assert text[pages[1].char_start : pages[1].char_end] == "Page two."
    assert pages[2].char_start == pages[2].char_end  # empty page keeps its place
    assert text[pages[3].char_start : pages[3].char_end] == "Page four."


def test_heading_remembers_its_page():
    b = TextBuilder()
    b.start_page(1)
    b.add("A", heading_level=1)
    b.start_page(2)
    b.add("B", heading_level=1)
    _, _, headings = b.build()
    assert [h.page for h in headings] == [1, 2]


def test_parsed_document_roundtrips_through_dict():
    doc = ParsedDocument(
        text="Hello\n\nWorld",
        title="T",
        pages=[PageSpan(1, 0, 5), PageSpan(2, 7, 12)],
        headings=[Heading(1, "Hello", 0, 1)],
        metadata={"parser": "x"},
        warnings=["w"],
    )
    assert ParsedDocument.from_dict(doc.to_dict()) == doc


def test_page_at_and_summary():
    doc = ParsedDocument(
        text="Hello\n\nWorld",
        pages=[PageSpan(1, 0, 5), PageSpan(2, 7, 12)],
        headings=[Heading(1, "Hello", 0, 1)],
        metadata={"parser": "x"},
    )
    assert doc.page_at(0) == 1
    assert doc.page_at(8) == 2
    assert doc.page_at(5) is None  # the separator belongs to no page
    assert doc.summary() == {
        "title": None,
        "characters": 12,
        "pages": 2,
        "headings": 1,
        "warnings": [],
        "parser": "x",
    }
