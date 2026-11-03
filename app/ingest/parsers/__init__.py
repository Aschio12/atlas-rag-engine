"""Format parsers (Phase 2.3). `parse()` is the single entry point."""

from __future__ import annotations

from app.ingest.files import Kind
from app.ingest.parsers.base import Heading, PageSpan, ParsedDocument, ParseError, TextBuilder
from app.ingest.parsers.html import parse_html
from app.ingest.parsers.plain import parse_csv, parse_markdown, parse_text

__all__ = [
    "Heading",
    "PageSpan",
    "ParseError",
    "ParsedDocument",
    "TextBuilder",
    "parse",
]


def parse(data: bytes, kind: Kind) -> ParsedDocument:
    """Parse a file of a known kind. Raises `ParseError` for unreadable files."""
    match kind:
        case Kind.TEXT:
            return parse_text(data)
        case Kind.MARKDOWN:
            return parse_markdown(data)
        case Kind.CSV:
            return parse_csv(data)
        case Kind.HTML:
            return parse_html(data)
    raise ParseError(f"No parser is available for {kind.value} files yet")
