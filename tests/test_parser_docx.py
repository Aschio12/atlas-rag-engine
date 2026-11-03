import io
import zipfile

import pytest
from docx import Document as WordDocument

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, parse


def build_docx() -> bytes:
    d = WordDocument()
    d.core_properties.title = "Employee Handbook"
    d.core_properties.author = "HR Team"
    d.add_heading("Employee Handbook", level=1)
    d.add_paragraph("Welcome to the company.")
    d.add_heading("Leave policy", level=2)
    d.add_paragraph("Staff get 25 days of leave.")
    d.add_paragraph("Carry over up to 5 days", style="List Bullet")
    d.add_paragraph("Request via the portal", style="List Bullet")
    table = d.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Level", "Days"), ("Junior", "20"), ("Senior", "25")]):
        table.cell(r, 0).text = a
        table.cell(r, 1).text = b
    d.add_heading("Contact", level=2)
    para = d.add_paragraph("Line one")
    para.add_run().add_break()
    para.add_run("line two\tafter tab")
    d.add_paragraph("   ")  # blank paragraph
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def doc():
    return parse(build_docx(), Kind.DOCX)


def test_metadata_and_title(doc):
    assert doc.title == "Employee Handbook"
    assert doc.metadata["author"] == "HR Team"
    assert doc.metadata["parser"] == "docx"
    assert doc.pages == []  # Word has no fixed pagination


def test_headings_levels_and_offsets(doc):
    assert [(h.level, h.text) for h in doc.headings] == [
        (1, "Employee Handbook"),
        (2, "Leave policy"),
        (2, "Contact"),
    ]
    for h in doc.headings:
        assert doc.text[h.char_start : h.char_start + len(h.text)] == h.text


def test_paragraphs_lists_and_tables_in_order(doc):
    t = doc.text
    assert t.index("Welcome") < t.index("Staff get 25") < t.index("- Carry over")
    assert "- Carry over up to 5 days\n- Request via the portal" in t
    assert "Level | Days\nJunior | 20\nSenior | 25" in t
    assert t.index("Senior | 25") < t.index("Contact")


def test_line_and_tab_breaks(doc):
    assert "Line one\nline two after tab" in doc.text


def test_blank_paragraphs_are_dropped(doc):
    assert "\n\n\n" not in doc.text


def test_tracked_deletions_are_excluded():
    d = WordDocument()
    p = d.add_paragraph("Keep this. ")
    xml = (
        '<w:del xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'w:id="1" w:author="x"><w:r><w:delText>Deleted text.</w:delText></w:r></w:del>'
    )
    from lxml import etree

    p._p.append(etree.fromstring(xml))
    buf = io.BytesIO()
    d.save(buf)
    assert parse(buf.getvalue(), Kind.DOCX).text == "Keep this."


def test_style_derived_from_a_heading_is_a_heading():
    d = WordDocument()
    base = d.styles["Heading 1"]
    custom = d.styles.add_style("Chapter Title", 1)
    custom.base_style = base
    d.add_paragraph("Chapter one", style="Chapter Title")
    d.add_paragraph("Body.")
    buf = io.BytesIO()
    d.save(buf)
    assert parse(buf.getvalue(), Kind.DOCX).headings[0].text == "Chapter one"


def test_long_paragraph_in_heading_style_is_not_a_heading():
    d = WordDocument()
    d.add_heading("x" * 400, level=1)
    d.add_paragraph("body")
    buf = io.BytesIO()
    d.save(buf)
    assert parse(buf.getvalue(), Kind.DOCX).headings == []


def test_empty_document_is_an_error():
    buf = io.BytesIO()
    WordDocument().save(buf)
    with pytest.raises(ParseError, match="no text"):
        parse(buf.getvalue(), Kind.DOCX)


def _zip(parts: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in parts.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_corrupt_package_is_an_error():
    with pytest.raises(ParseError, match="corrupt"):
        parse(b"PK\x03\x04 definitely not a zip", Kind.DOCX)


def test_missing_document_part_is_an_error():
    with pytest.raises(ParseError, match="missing a required part"):
        parse(_zip({"word/styles.xml": "<x/>"}), Kind.DOCX)


def test_malformed_xml_is_an_error():
    with pytest.raises(ParseError, match="corrupt"):
        parse(_zip({"word/document.xml": "<w:document><unclosed>"}), Kind.DOCX)


def test_xml_entity_attacks_are_rejected():
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&lol2;</w:t></w:r></w:p></w:body></w:document>"
    )
    with pytest.raises(ParseError, match="disallowed XML"):
        parse(_zip({"word/document.xml": bomb}), Kind.DOCX)


def test_external_entities_are_rejected(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    xxe = (
        f'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file://{secret}">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>"
    )
    with pytest.raises(ParseError, match="disallowed XML"):
        parse(_zip({"word/document.xml": xxe}), Kind.DOCX)


def test_zip_bombs_are_rejected():
    ratio_bomb = _zip({"word/document.xml": "<a>" + "0" * (60 * 1024 * 1024) + "</a>"})
    assert len(ratio_bomb) < 1024 * 1024  # tiny on disk, huge when expanded
    with pytest.raises(ParseError, match="compression bomb"):
        parse(ratio_bomb, Kind.DOCX)
