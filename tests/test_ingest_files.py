import io
import zipfile

import pytest

from app.ingest.files import (
    Kind,
    UnsupportedFile,
    decode_text,
    detect_kind,
    sanitize_filename,
    sha256_hex,
    title_from_filename,
)


def _zip(*names: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, "<x/>")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("data", "filename", "content_type", "expected"),
    [
        (b"%PDF-1.7\n...", "a.pdf", None, Kind.PDF),
        (b"%PDF-1.7\n...", None, None, Kind.PDF),
        (b"\n  %PDF-1.4", "scan", "application/octet-stream", Kind.PDF),
        (_zip("word/document.xml"), "a.docx", None, Kind.DOCX),
        (_zip("ppt/presentation.xml"), "deck.pptx", None, Kind.PPTX),
        (_zip("word/document.xml"), None, None, Kind.DOCX),
        (b"# Title\n\ntext", "notes.md", None, Kind.MARKDOWN),
        (b"a,b\n1,2\n", "t.csv", None, Kind.CSV),
        (b"a\tb\n", "t.tsv", None, Kind.CSV),
        (b"<html><body>x</body></html>", "p.htm", None, Kind.HTML),
        (b"<!DOCTYPE html><p>x", None, None, Kind.HTML),
        (b"<p>x</p>", None, "text/html; charset=utf-8", Kind.HTML),
        (b"plain words", "n.txt", None, Kind.TEXT),
        (b"plain words", None, None, Kind.TEXT),
        (b"plain words", "README", None, Kind.TEXT),
        (b"x,y", None, "text/csv", Kind.CSV),
        ("héllo".encode("utf-16"), "u.txt", None, Kind.TEXT),
    ],
)
def test_detects_kind(data, filename, content_type, expected):
    assert detect_kind(data, filename, content_type) == expected


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"", "empty.txt"),
        (b"   \n ", "blank.txt"),
        (b"MZ\x90\x00\x03\x00\x00\x00", "evil.pdf"),  # exe renamed to pdf
        (b"plain text", "fake.pdf"),
        (b"plain text", "fake.docx"),
        (b"%PDF-1.7", "report.docx"),  # real PDF, wrong name
        (b"%PDF-1.7", "report.txt"),
        (_zip("word/document.xml"), "deck.pptx"),
        (_zip("xl/workbook.xml"), "sheet.xlsx"),
        (b"PK\x03\x04garbage", "a.docx"),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "image.png"),
        (b"abc\x00def", "blob.bin"),
        (b"hello", "archive.exe"),
    ],
)
def test_rejects(data, filename):
    with pytest.raises(UnsupportedFile):
        detect_kind(data, filename)


def test_text_with_bom_is_not_mistaken_for_binary():
    assert detect_kind("hi".encode("utf-16"), "a.txt") == Kind.TEXT


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\bob\\cv.docx", "cv.docx"),
        ("a/b/c.txt", "c.txt"),
        ("na\x00me\x07.txt", "name.txt"),
        ("what?<is>:this|.md", "whatisthis.md"),
        ("  ..hidden.. ", "hidden"),
        ("", "document"),
        (None, "document"),
        ("...", "document"),
        ("\u202eexe.txt", "exe.txt"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_bounds_length_and_keeps_extension():
    out = sanitize_filename("x" * 500 + ".pdf")
    assert len(out) <= 200
    assert out.endswith(".pdf")


def test_title_from_filename():
    assert title_from_filename("Q3_sales-report.final.pdf") == "Q3 sales-report.final"
    assert title_from_filename("notes") == "notes"
    assert title_from_filename(".env") == ".env"


def test_sha256_matches_known_vector():
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_decode_text_variants():
    assert decode_text("héllo".encode()) == ("héllo", "utf-8")
    assert decode_text(b"\xef\xbb\xbfhi") == ("hi", "utf-8")
    assert decode_text("héllo".encode("utf-16")) == ("héllo", "utf-16")
    assert decode_text("café".encode("cp1252")) == ("café", "cp1252")
    text, enc = decode_text(b"\x81\x8d")  # undefined in cp1252: replaced, never raises
    assert enc == "cp1252"
    assert text
