import pytest

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, parse


def line_at(doc, offset):
    end = doc.text.find("\n", offset)
    return doc.text[offset : end if end != -1 else len(doc.text)]


# --- text ------------------------------------------------------------------------------------


def test_text_is_normalised_and_keeps_indentation():
    doc = parse(b"Title\r\n\r\n\r\n\r\n    indented  line\t\n", Kind.TEXT)
    assert doc.text == "Title\n\n    indented line"
    assert doc.metadata == {"parser": "text", "encoding": "utf-8"}
    assert doc.pages == [] and doc.headings == []


def test_text_decodes_utf16_and_cp1252():
    assert parse("héllo wörld".encode("utf-16"), Kind.TEXT).text == "héllo wörld"
    doc = parse("café au lait".encode("cp1252"), Kind.TEXT)
    assert doc.text == "café au lait"
    assert doc.metadata["encoding"] == "cp1252"


def test_text_without_content_is_an_error():
    with pytest.raises(ParseError, match="no readable text"):
        parse(b" \x00\x07 \n", Kind.TEXT)


# --- markdown --------------------------------------------------------------------------------

MD = """---
title: Refund Policy
author: Support Team
---

# Refunds

Intro paragraph.

## Eligibility

Items can be returned within *14 days*.

```
# not a heading, it is code
```

Setext Title
============

Sub **bold** heading
--------------------

### Contact [us](https://example.com) ###
"""


def test_markdown_front_matter_title_and_author():
    doc = parse(MD.encode(), Kind.MARKDOWN)
    assert doc.title == "Refund Policy"
    assert doc.metadata["author"] == "Support Team"
    assert "---" not in doc.text.split("\n")[0]
    assert doc.text.startswith("# Refunds")


def test_markdown_headings_levels_text_and_offsets():
    doc = parse(MD.encode(), Kind.MARKDOWN)
    got = [(h.level, h.text) for h in doc.headings]
    assert got == [
        (1, "Refunds"),
        (2, "Eligibility"),
        (1, "Setext Title"),
        (2, "Sub bold heading"),
        (3, "Contact us"),
    ]
    for h in doc.headings:
        assert line_at(doc, h.char_start).lstrip("# ").startswith(h.text.split()[0])


def test_markdown_ignores_headings_inside_code_fences():
    doc = parse(MD.encode(), Kind.MARKDOWN)
    assert "not a heading, it is code" not in [h.text for h in doc.headings]


def test_markdown_title_falls_back_to_first_h1():
    doc = parse(b"intro\n\n# Real Title\n\ntext", Kind.MARKDOWN)
    assert doc.title == "Real Title"


def test_markdown_without_headings_has_no_title():
    doc = parse(b"just a paragraph", Kind.MARKDOWN)
    assert doc.title is None and doc.headings == []


def test_markdown_crlf_and_unclosed_front_matter():
    doc = parse(b"---\r\ntitle: x\r\n\r\n# H\r\n", Kind.MARKDOWN)  # no closing ---: kept as text
    assert doc.headings[0].text == "H"
    assert "title: x" in doc.text


def test_empty_markdown_after_front_matter_is_an_error():
    with pytest.raises(ParseError):
        parse(b"---\ntitle: x\n---\n", Kind.MARKDOWN)


# --- csv -------------------------------------------------------------------------------------


def test_csv_rows_become_labelled_lines():
    doc = parse(b"name,plan,price\nAda,Pro,29\nLin,Free,0\n", Kind.CSV)
    assert doc.text == "name: Ada | plan: Pro | price: 29\nname: Lin | plan: Free | price: 0"
    assert doc.metadata["rows"] == 2
    assert doc.metadata["columns"] == ["name", "plan", "price"]
    assert doc.metadata["has_header"] is True


@pytest.mark.parametrize(("delimiter", "name"), [(";", "semicolon"), ("\t", "tab"), ("|", "pipe")])
def test_csv_detects_delimiters(delimiter, name):
    raw = f"a{delimiter}b\n1{delimiter}2\n3{delimiter}4\n".encode()
    doc = parse(raw, Kind.CSV)
    assert doc.text == "a: 1 | b: 2\na: 3 | b: 4", name


def test_csv_handles_quotes_embedded_commas_and_newlines():
    raw = b'q,a\n"Hello, world","line1\nline2"\n'
    doc = parse(raw, Kind.CSV)
    assert doc.text.startswith("q: Hello, world | a: line1\nline2")


def test_csv_without_a_header_uses_column_numbers():
    doc = parse(b"1,2\n3,4\n5,6\n", Kind.CSV)  # an all-numeric first row is data, not a header
    assert doc.metadata["has_header"] is False
    assert doc.metadata["rows"] == 3
    assert doc.text.splitlines()[0] == "column 1: 1 | column 2: 2"


def test_csv_duplicate_header_names_are_not_a_header():
    doc = parse(b"x,x\n1,2\n", Kind.CSV)
    assert doc.metadata["has_header"] is False
    assert "column 1: x" in doc.text


def test_csv_skips_blank_rows_and_empty_cells_and_warns_on_ragged_rows():
    doc = parse(b"a,b,c\n\n1,,3\n4,5\n", Kind.CSV)
    assert doc.text == "a: 1 | c: 3\na: 4 | b: 5"
    assert any("different number of columns" in w for w in doc.warnings)


@pytest.mark.parametrize("raw", [b"", b"\n\n", b",,\n,,\n"])
def test_empty_csv_is_an_error(raw):
    with pytest.raises(ParseError):
        parse(raw, Kind.CSV)


def test_csv_header_only_is_an_error():
    with pytest.raises(ParseError):
        parse(b"a,b,c\n", Kind.CSV)


# --- dispatcher ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(Kind))
def test_every_kind_has_a_parser(kind):
    """Garbage input must reach the real parser and fail as a ParseError, never 'No parser'."""
    try:
        parse(b"x", kind)
    except ParseError as e:
        assert "No parser" not in str(e)


# --- code -----------------------------------------------------------------------------------


def test_code_keeps_indentation_and_reports_the_language():
    src = b"def greet(name):\n    if name:\n        return f'hi {name}'\n    return 'hi'\n"
    doc = parse(src, Kind.CODE, filename="greet.py")
    assert doc.text == src.decode().rstrip("\n")
    assert "    if name:\n        return" in doc.text
    assert doc.metadata == {"parser": "code", "encoding": "utf-8", "code_language": "python"}
    assert doc.headings == [] and doc.pages == []


def test_code_without_a_known_extension_has_no_language():
    doc = parse(b"echo hi\n", Kind.CODE, filename="mystery.zzz")
    assert "code_language" not in doc.metadata
    assert "code_language" not in parse(b"echo hi\n", Kind.CODE).metadata


def test_empty_code_is_an_error():
    with pytest.raises(ParseError):
        parse(b"\n  \n", Kind.CODE, filename="a.py")
