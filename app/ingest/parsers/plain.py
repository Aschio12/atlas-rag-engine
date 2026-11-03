"""Plain text, Markdown and CSV."""

from __future__ import annotations

import csv
import io
import re

from app.ingest.files import decode_text
from app.ingest.parsers.base import Heading, ParsedDocument, ParseError, TextBuilder, normalize

# --- plain text ----------------------------------------------------------------------------


def parse_text(data: bytes) -> ParsedDocument:
    raw, encoding = decode_text(data)
    text = normalize(raw, keep_indent=True)
    if not text:
        raise ParseError("The file contains no readable text")
    return ParsedDocument(text=text, metadata={"parser": "text", "encoding": encoding})


# --- markdown ------------------------------------------------------------------------------

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)
_INLINE = re.compile(r"(\*\*|__|\*|_|`)(.+?)\1")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def _plain_heading(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _INLINE.sub(r"\2", text)
    return text.strip()


def _front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t", "-")):
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[m.end() :]


def parse_markdown(data: bytes) -> ParsedDocument:
    raw, encoding = decode_text(data)
    meta, body = _front_matter(raw.replace("\r\n", "\n").replace("\r", "\n"))
    text = normalize(body, keep_indent=True)
    if not text:
        raise ParseError("The file contains no readable text")

    headings: list[Heading] = []
    in_fence = False
    offset = 0
    previous: tuple[str, int] | None = None  # (line, offset) of a possible setext title
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            if m := _ATX.match(line):
                headings.append(Heading(len(m.group(1)), _plain_heading(m.group(2)), offset))
            elif previous and (s := _SETEXT.match(line)) and previous[0].strip():
                level = 1 if s.group(1).startswith("=") else 2
                headings.append(Heading(level, _plain_heading(previous[0]), previous[1]))
        previous = (line, offset)
        offset += len(line) + 1

    title = meta.get("title") or next((h.text for h in headings if h.level == 1), None)
    metadata = {"parser": "markdown", "encoding": encoding}
    if meta.get("author"):
        metadata["author"] = meta["author"]
    return ParsedDocument(text=text, title=title, headings=headings, metadata=metadata)


# --- csv -----------------------------------------------------------------------------------

MAX_CSV_ROWS = 200_000


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", ""))
    except ValueError:
        return False
    return True


def _dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def parse_csv(data: bytes) -> ParsedDocument:
    """Each row becomes one line, `Header: value | Header: value`, so a chunk of rows stays
    understandable (and searchable by column name) without the header row."""
    raw, encoding = decode_text(data)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    csv.field_size_limit(10 * 1024 * 1024)
    try:
        rows = [
            r
            for r in csv.reader(io.StringIO(raw), _dialect(raw[:8192]))
            if any(c.strip() for c in r)
        ]
    except csv.Error as e:
        raise ParseError(f"The CSV file is malformed ({e})") from None
    if not rows:
        raise ParseError("The CSV file is empty")

    if len(rows) == 1:
        raise ParseError("The CSV file has a single row and no data to index")
    warnings: list[str] = []
    header = [c.strip() for c in rows[0]]
    has_header = (
        len(set(header)) == len(header) and all(header) and not all(_is_number(h) for h in header)
    )
    body = rows[1:] if has_header else rows
    if not has_header:
        header = [f"column {i + 1}" for i in range(max(len(r) for r in rows))]
    if len(body) > MAX_CSV_ROWS:
        warnings.append(f"Only the first {MAX_CSV_ROWS:,} of {len(body):,} rows were read")
        body = body[:MAX_CSV_ROWS]

    builder = TextBuilder()
    ragged = 0
    for row in body:
        if len(row) != len(header):
            ragged += 1
        cells = [
            f"{header[i] if i < len(header) else f'column {i + 1}'}: {c.strip()}"
            for i, c in enumerate(row)
            if c.strip()
        ]
        builder.add(" | ".join(cells), sep="\n")
    if ragged:
        warnings.append(f"{ragged} row(s) had a different number of columns than the header")
    text, _, _ = builder.build()
    if not text:
        raise ParseError("The CSV file has headers but no data")
    return ParsedDocument(
        text=text,
        metadata={
            "parser": "csv",
            "encoding": encoding,
            "rows": len(body),
            "columns": header,
            "has_header": has_header,
        },
        warnings=warnings,
    )
