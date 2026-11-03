import io

import pytest
from pptx import Presentation
from pptx.util import Inches

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, parse


def build_pptx() -> bytes:
    prs = Presentation()
    prs.core_properties.title = "Q3 Business Review"
    prs.core_properties.author = "Finance"

    s1 = prs.slides.add_slide(prs.slide_layouts[0])  # title slide
    s1.shapes.title.text = "Q3 Business Review"
    s1.placeholders[1].text = "Prepared by Finance"

    s2 = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    s2.shapes.title.text = "Revenue"
    lower = s2.shapes.add_textbox(Inches(1), Inches(5), Inches(4), Inches(1))
    lower.text_frame.text = "Bottom text"
    upper = s2.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    upper.text_frame.text = "Top text"
    table = s2.shapes.add_table(2, 2, Inches(1), Inches(3), Inches(4), Inches(1)).table
    for (r, c), v in {(0, 0): "Region", (0, 1): "Sales", (1, 0): "EMEA", (1, 1): "$4M"}.items():
        table.cell(r, c).text = v
    s2.notes_slide.notes_text_frame.text = "Mention the EMEA launch."

    s3 = prs.slides.add_slide(prs.slide_layouts[6])  # blank, no title
    box = s3.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    box.text_frame.text = "Appendix without a title"

    prs.slides.add_slide(prs.slide_layouts[6])  # a completely empty slide
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def doc():
    return parse(build_pptx(), Kind.PPTX)


def test_metadata_and_title(doc):
    assert doc.title == "Q3 Business Review"
    assert doc.metadata["slides"] == 4
    assert doc.metadata["slides_with_notes"] == 1
    assert doc.metadata["author"] == "Finance"


def test_each_slide_is_a_page_and_pages_cover_the_text(doc):
    assert [p.page for p in doc.pages] == [1, 2, 3, 4]
    s2 = next(p for p in doc.pages if p.page == 2)
    page_text = doc.text[s2.char_start : s2.char_end]
    assert page_text.startswith("Revenue")
    assert "Mention the EMEA launch." in page_text
    assert "Appendix" not in page_text
    empty = doc.pages[3]
    assert empty.char_start == empty.char_end


def test_slide_titles_are_headings_with_pages(doc):
    assert [(h.level, h.text, h.page) for h in doc.headings] == [
        (1, "Q3 Business Review", 1),
        (1, "Revenue", 2),
    ]
    for h in doc.headings:
        assert doc.text[h.char_start : h.char_start + len(h.text)] == h.text


def test_reading_order_is_top_to_bottom(doc):
    assert doc.text.index("Top text") < doc.text.index("Bottom text")


def test_tables_and_notes_are_included(doc):
    assert "Region | Sales\nEMEA | $4M" in doc.text
    assert "Speaker notes: Mention the EMEA launch." in doc.text


def test_slide_without_a_title_still_contributes_text(doc):
    assert "Appendix without a title" in doc.text
    assert doc.page_at(doc.text.index("Appendix")) == 3


def test_slide_order_follows_the_presentation_not_the_file_names():
    prs = Presentation()
    for name in ("first", "second", "third"):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = name
    ids = prs.slides._sldIdLst
    ids.insert(0, ids[2])  # move "third" to the front
    buf = io.BytesIO()
    prs.save(buf)
    doc = parse(buf.getvalue(), Kind.PPTX)
    assert [h.text for h in doc.headings] == ["third", "first", "second"]


def test_empty_presentation_is_an_error():
    buf = io.BytesIO()
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    prs.save(buf)
    with pytest.raises(ParseError, match="no text"):
        parse(buf.getvalue(), Kind.PPTX)


def test_presentation_without_slides_is_an_error():
    buf = io.BytesIO()
    Presentation().save(buf)
    with pytest.raises(ParseError, match="no slides"):
        parse(buf.getvalue(), Kind.PPTX)


def test_corrupt_deck_is_an_error():
    with pytest.raises(ParseError, match="corrupt"):
        parse(b"PK\x03\x04 nope", Kind.PPTX)
