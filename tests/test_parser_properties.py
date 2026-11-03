"""Invariants every parser must keep, checked on real files and on hostile/random input."""

import random
import re

import pdfgen
import pytest
from test_ingest_pipeline import CASES

from app.ingest.files import Kind
from app.ingest.parsers import ParsedDocument, ParseError, parse
from app.ingest.parsers.base import normalize

KINDS = {"txt": Kind.TEXT, "md": Kind.MARKDOWN, "csv": Kind.CSV, "html": Kind.HTML,
         "docx": Kind.DOCX, "pptx": Kind.PPTX, "pdf": Kind.PDF}  # fmt: skip
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​﻿\r]")


def check_invariants(doc: ParsedDocument) -> None:
    text = doc.text
    assert text and text == text.strip("\n")
    assert not CONTROL.search(text)
    assert "\n\n\n" not in text
    assert all(line == line.rstrip(" ") for line in text.split("\n"))
    assert normalize(text, keep_indent=True) == text  # already canonical

    previous_end = 0
    for span in doc.pages:
        assert 0 <= span.char_start <= span.char_end <= len(text)
        assert span.char_start >= previous_end  # ordered, non-overlapping
        previous_end = span.char_end
    starts = [h.char_start for h in doc.headings]
    assert starts == sorted(starts)
    for h in doc.headings:
        assert 1 <= h.level <= 6
        assert h.text.strip() == h.text and h.text
        assert 0 <= h.char_start < len(text)
        if h.page is not None:
            assert h.page in {p.page for p in doc.pages}
    assert ParsedDocument.from_dict(doc.to_dict()) == doc  # lossless serialisation


@pytest.mark.parametrize("kind", list(CASES))
def test_real_files_satisfy_the_invariants(kind):
    _, data, *_ = CASES[kind]
    check_invariants(parse(data, KINDS[kind]))


@pytest.mark.parametrize("kind", list(Kind))
def test_random_bytes_never_crash_a_parser(kind):
    rng = random.Random(f"bytes-{kind}")
    for _ in range(150):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 400)))
        try:
            check_invariants(parse(data, kind))
        except ParseError:
            pass


@pytest.mark.parametrize("kind", ["docx", "pptx", "pdf"])
def test_truncated_files_fail_cleanly(kind):
    _, data, *_ = CASES[kind]
    rng = random.Random(f"trunc-{kind}")
    for cut in sorted({rng.randrange(1, len(data)) for _ in range(25)}):
        try:
            check_invariants(parse(data[:cut], KINDS[kind]))
        except ParseError:
            pass


@pytest.mark.parametrize("kind", ["docx", "pptx", "pdf", "html", "md"])
def test_corrupted_files_fail_cleanly(kind):
    _, data, *_ = CASES[kind]
    rng = random.Random(f"flip-{kind}")
    for _ in range(25):
        broken = bytearray(data)
        for _ in range(rng.randrange(1, 12)):
            broken[rng.randrange(len(broken))] = rng.randrange(256)
        try:
            check_invariants(parse(bytes(broken), KINDS[kind]))
        except ParseError:
            pass


MARKUP = ["# ", "## ", "\n", "\n\n", "\r\n", "  ", "\t", "word", "Äpfel", "<p>", "</p>", "<h2>",
          "</h2>", "<li>", "<td>", "<script>", "</script>", "---", "===", "```", "|", ",", '"',
          "​", "\x00", "&amp;", "<", ">", "- ", "1. "]  # fmt: skip


@pytest.mark.parametrize("kind", [Kind.TEXT, Kind.MARKDOWN, Kind.CSV, Kind.HTML])
def test_random_markup_soup_keeps_the_invariants(kind):
    rng = random.Random(f"soup-{kind}")
    for _ in range(300):
        soup = "".join(rng.choice(MARKUP) for _ in range(rng.randrange(1, 40)))
        try:
            check_invariants(parse(soup.encode(), kind))
        except ParseError:
            pass


def test_parsing_is_deterministic():
    for kind, (_, data, *_) in CASES.items():
        assert parse(data, KINDS[kind]).to_dict() == parse(data, KINDS[kind]).to_dict(), kind


def test_unicode_survives_in_every_text_format():
    word = "Zażółć gęślą jaźń — 日本語 — 😀"
    for kind, payload in [
        (Kind.TEXT, word),
        (Kind.MARKDOWN, f"# {word}\n\n{word}"),
        (Kind.CSV, f"a,b\n{word},x\n"),
        (Kind.HTML, f"<h1>{word}</h1><p>{word}</p>"),
    ]:
        doc = parse(payload.encode(), kind)
        assert word in doc.text, kind
        check_invariants(doc)


def test_generated_pdfs_of_many_pages_stay_consistent():
    doc = parse(pdfgen.report_pdf(), Kind.PDF)
    for span in doc.pages:
        chunk = doc.text[span.char_start : span.char_end]
        assert chunk == normalize(chunk, keep_indent=True)
