"""Format parsers (Phase 2.3). `parse()` is the single entry point."""

from __future__ import annotations

from app.ingest.files import Kind
from app.ingest.parsers.base import Heading, PageSpan, ParsedDocument, ParseError, TextBuilder
from app.ingest.parsers.docx import parse_docx
from app.ingest.parsers.html import parse_html
from app.ingest.parsers.ocr import OcrEngine, default_ocr
from app.ingest.parsers.pdf import parse_pdf
from app.ingest.parsers.plain import parse_csv, parse_markdown, parse_text
from app.ingest.parsers.pptx import parse_pptx

__all__ = [
    "Heading",
    "OcrEngine",
    "PageSpan",
    "ParseError",
    "ParsedDocument",
    "TextBuilder",
    "default_ocr",
    "parse",
]


def parse(data: bytes, kind: Kind, *, ocr: OcrEngine | None = None) -> ParsedDocument:
    """Parse a file of a known kind. Raises `ParseError` for unreadable files.

    `ocr` is used for scanned PDF pages; pass `default_ocr()` to use Tesseract when installed.
    """
    match kind:
        case Kind.TEXT:
            return parse_text(data)
        case Kind.MARKDOWN:
            return parse_markdown(data)
        case Kind.CSV:
            return parse_csv(data)
        case Kind.HTML:
            return parse_html(data)
        case Kind.DOCX:
            return parse_docx(data)
        case Kind.PPTX:
            return parse_pptx(data)
        case Kind.PDF:
            return parse_pdf(data, ocr=ocr)
    raise ParseError(f"No parser is available for {kind.value} files yet")
