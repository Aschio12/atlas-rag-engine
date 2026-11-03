import pdfgen
import pytest

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, parse


@pytest.fixture(scope="module")
def doc():
    return parse(pdfgen.report_pdf(), Kind.PDF)


def test_pages_and_metadata(doc):
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert doc.metadata["page_count"] == 3
    assert doc.metadata["parser"] == "pdf"
    assert doc.metadata["author"] == "ACME Finance"
    assert doc.title == "Annual Report 2025"


def test_page_spans_partition_the_text(doc):
    for span in doc.pages:
        assert 0 <= span.char_start <= span.char_end <= len(doc.text)
    for a, b in zip(doc.pages, doc.pages[1:], strict=False):
        assert a.char_end <= b.char_start
    assert "Regional breakdown" in doc.text[doc.pages[1].char_start : doc.pages[1].char_end]


def test_headings_found_by_font_size_with_levels_and_pages(doc):
    got = [(h.level, h.text, h.page) for h in doc.headings]
    assert (1, "Annual Report", 1) in got
    assert (2, "Financial results", 1) in got
    assert any(t == "Regional breakdown" and lvl == 2 for lvl, t, _ in got)
    levels = {t: lvl for lvl, t, _ in got}
    assert levels["Annual Report"] < levels["Financial results"] <= levels["Outlook"]
    for h in doc.headings:
        assert doc.text[h.char_start : h.char_start + len(h.text)] == h.text


def test_body_text_is_not_mistaken_for_headings(doc):
    assert not any("Our company" in h.text for h in doc.headings)


def test_running_headers_and_page_numbers_are_removed(doc):
    assert "ACME Corp Confidential" not in doc.text
    assert "Page 1" not in doc.text and "Page 3" not in doc.text
    assert doc.metadata["removed_running_lines"] == 6


def test_paragraph_lines_are_joined_and_hyphenation_repaired(doc):
    assert "Our company continued to grow steadily throughout the year, expanding" in doc.text
    assert "\nOur company continued to grow steadily throughout the year,\n" not in doc.text
    assert "international offerings remains strong." in doc.text


def test_bullets_are_kept_on_consecutive_lines(doc):
    assert "- Record quarterly revenue\n- New offices opened\n- Headcount up 8 percent" in doc.text
    assert "high.\n\n- Record quarterly revenue" in doc.text  # the list is its own block


def test_ruled_tables_become_rows_in_reading_position(doc):
    assert "Region | Revenue | Growth\nEMEA | $4.0M | 12%\nAPAC | $2.5M | 9%" in doc.text
    assert doc.text.index("Regional breakdown") < doc.text.index("EMEA | $4.0M")
    assert doc.text.index("EMEA | $4.0M") < doc.text.index("The table above summarises")
    # the table text is not also emitted as scrambled paragraph text
    assert doc.text.count("EMEA") == 1


def test_two_column_pages_are_read_column_by_column(doc):
    section = doc.text[doc.pages[2].char_start :]
    last_left = section.rindex("LEFTCOLUMN")
    first_right = section.index("RIGHTCOLUMN")
    assert last_left < first_right  # whole left column before any of the right
    assert section.index("Two column section") < section.index("LEFTCOLUMN")


def test_title_falls_back_to_first_h1_when_metadata_is_junk():
    d = parse(pdfgen.report_pdf(title="Microsoft Word - draft.docx"), Kind.PDF)
    assert d.title == "Annual Report"


def test_roundtrips_through_json(doc):
    from app.ingest.parsers import ParsedDocument

    assert ParsedDocument.from_dict(doc.to_dict()) == doc


def test_garbage_pdf_is_a_clean_error():
    with pytest.raises(ParseError, match="could not be read"):
        parse(b"%PDF-1.4\nthis is not really a pdf\n%%EOF", Kind.PDF)


def test_truncated_pdf_is_a_clean_error():
    with pytest.raises(ParseError):
        parse(pdfgen.report_pdf()[:300], Kind.PDF)


def test_blank_pdf_is_an_error_without_ocr():
    with pytest.raises(ParseError, match="no extractable text"):
        parse(pdfgen.blank_pdf(), Kind.PDF)
