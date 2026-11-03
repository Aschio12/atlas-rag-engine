"""Layout-aware PDF parsing with pdfplumber.

What "layout-aware" means here:
* words are regrouped into lines and paragraphs from their coordinates (line gaps, hyphenation);
* **two-column pages** are detected from the empty gutter and read column by column, with
  full-width titles and footnotes kept in their place;
* **headings** are found by font size (and bold for the smallest level) relative to the body text;
* **ruled tables** are extracted as `cell | cell` rows instead of scrambled text;
* **running headers/footers** and page numbers repeated on most pages are removed;
* **scanned pages** (no text layer, only an image) go through an optional OCR engine.
"""

from __future__ import annotations

import io
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import pdfplumber

from app.ingest.parsers.base import ParsedDocument, ParseError, TextBuilder
from app.ingest.parsers.ocr import OcrEngine

log = logging.getLogger("atlas.parsers.pdf")

MAX_PAGES = 1_500
HEADING_RATIO = 1.18  # a line this much larger than body text is a heading
MAX_HEADING_CHARS = 160
PARAGRAPH_GAP = 0.45  # blank space between lines, as a fraction of the font size
RUNNING_ZONE = 0.08  # top/bottom share of the page searched for running headers/footers
_BULLET = re.compile(r"^([•▪◦·●■‣\-–—*]|\d{1,3}[.)]|\(?[a-z][.)])\s+(?=\S)")
_SYMBOL_BULLETS = set("•▪◦·●■‣-–—*")
_JUNK_TITLE = re.compile(r"^(microsoft word - |untitled|\S+\.(docx?|pptx?|tex|indd|pdf)$)", re.I)


@dataclass
class _Row:
    kind: str  # "line" | "table"
    region: str  # "full" | "left" | "right"
    top: float
    bottom: float
    x0: float
    text: str
    size: float = 0.0
    bold: bool = False


@dataclass
class _Page:
    number: int
    height: float
    rows: list[_Row] = field(default_factory=list)  # reading order
    ocr_text: str | None = None


# --- geometry -------------------------------------------------------------------------------


def _cluster_rows(words: list[dict]) -> list[list[dict]]:
    """Group words that sit on the same baseline."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows:
            ref = rows[-1][0]
            tol = max(2.0, 0.45 * min(w["size"], ref["size"]))
            if abs(w["top"] - ref["top"]) <= tol or (
                # a superscript/subscript or tall glyph overlapping the row vertically
                w["top"] < max(x["bottom"] for x in rows[-1]) - 0.6 * w["size"]
                and abs(w["bottom"] - ref["bottom"]) <= tol
            ):
                rows[-1].append(w)
                continue
        rows.append([w])
    return rows


def _find_gutter(words: list[dict], width: float) -> tuple[float, float] | None:
    """The empty vertical band between two text columns, or None for single-column pages."""
    n = len(words)
    if n < 40:
        return None
    lo, hi = width * 0.3, width * 0.7
    step = 2.0
    allowed = max(1, int(0.03 * n))  # a few full-width lines may cross the gutter
    runs: list[tuple[float, float]] = []
    start: float | None = None
    x = lo
    while x <= hi:
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        if crossing <= allowed:
            start = x if start is None else start
        elif start is not None:
            runs.append((start, x - step))
            start = None
        x += step
    if start is not None:
        runs.append((start, hi))
    runs = [r for r in runs if r[1] - r[0] >= 10.0]
    if not runs:
        return None
    a, b = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - width / 2))
    mid = (a + b) / 2
    left = sum(1 for w in words if w["x1"] <= mid)
    right = sum(1 for w in words if w["x0"] >= mid)
    if min(left, right) < 0.2 * n:
        return None
    return a, b


def _line_text(words: list[dict]) -> str:
    return " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"]))


def _is_bold(words: list[dict]) -> bool:
    return all(re.search(r"bold|black|heavy", w.get("fontname", ""), re.I) for w in words)


def _row_from_words(words: list[dict], region: str) -> _Row:
    sizes = [w["size"] for w in words for _ in w["text"]]  # weighted by characters
    return _Row(
        kind="line",
        region=region,
        top=min(w["top"] for w in words),
        bottom=max(w["bottom"] for w in words),
        x0=min(w["x0"] for w in words),
        text=_line_text(words),
        size=round(statistics.median(sizes), 1),
        bold=_is_bold(words),
    )


def _inside(word: dict, boxes: list[tuple[float, float, float, float]]) -> bool:
    cx, cy = (word["x0"] + word["x1"]) / 2, (word["top"] + word["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in boxes)


def _extract_tables(page) -> tuple[list[_Row], list[tuple[float, float, float, float]]]:
    rows, boxes = [], []
    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001 - table detection is best-effort
        log.debug("table detection failed on page %s", page.page_number, exc_info=True)
        return rows, boxes
    for t in tables:
        data = [[re.sub(r"\s+", " ", c or "").strip() for c in r] for r in (t.extract() or [])]
        data = [r for r in data if any(r)]
        if len(data) < 2 or max((len(r) for r in data), default=0) < 2:
            continue  # a single box is not a table
        x0, top, x1, bottom = t.bbox
        boxes.append((x0, top, x1, bottom))
        text = "\n".join(" | ".join(c for c in r if c) for r in data)
        rows.append(_Row("table", "full", top, bottom, x0, text))
    return rows, boxes


def _read_page(page, number: int, ocr: OcrEngine | None, warnings: list[str]) -> _Page:
    out = _Page(number=number, height=float(page.height))
    table_rows, boxes = _extract_tables(page)
    words = [
        w
        for w in page.extract_words(
            extra_attrs=["size", "fontname"], keep_blank_chars=False, use_text_flow=False
        )
        if w["text"].strip() and not _inside(w, boxes)
    ]
    if not words and not table_rows:
        if page.images:  # a picture and no text layer: a scan
            if ocr is None:
                warnings.append(f"Page {number} has no text layer and OCR is not available")
            else:
                out.ocr_text = _ocr_page(page, number, ocr, warnings)
        return out

    gutter = _find_gutter(words, float(page.width))
    entries: list[_Row] = list(table_rows)
    for row_words in _cluster_rows(words):
        if gutter is None or any(w["x0"] < gutter[1] and w["x1"] > gutter[0] for w in row_words):
            entries.append(_row_from_words(row_words, "full"))
            continue
        mid = (gutter[0] + gutter[1]) / 2
        left = [w for w in row_words if w["x1"] <= mid]
        right = [w for w in row_words if w["x0"] >= mid]
        if left:
            entries.append(_row_from_words(left, "left"))
        if right:
            entries.append(_row_from_words(right, "right"))
    out.rows = _reading_order(entries)
    return out


def _reading_order(entries: list[_Row]) -> list[_Row]:
    """Top to bottom, but columns are read whole (left, then right) between full-width rows."""
    ordered: list[_Row] = []
    left: list[_Row] = []
    right: list[_Row] = []

    def flush() -> None:
        ordered.extend(sorted(left, key=lambda r: r.top))
        ordered.extend(sorted(right, key=lambda r: r.top))
        left.clear()
        right.clear()

    for row in sorted(entries, key=lambda r: (r.top, r.x0)):
        if row.region == "full":
            flush()
            ordered.append(row)
        elif row.region == "left":
            left.append(row)
        else:
            right.append(row)
    flush()
    return ordered


def _ocr_page(page, number: int, ocr: OcrEngine, warnings: list[str]) -> str | None:
    try:
        image = page.to_image(resolution=200).original
        return ocr.recognize(image).strip() or None
    except Exception:  # noqa: BLE001 - OCR is optional; a failure must not sink the document
        log.warning("OCR failed on page %s", number, exc_info=True)
        warnings.append(f"OCR failed on page {number}")
        return None


# --- document-level analysis ----------------------------------------------------------------


def _body_size(pages: list[_Page]) -> float:
    hist: Counter[float] = Counter()
    for page in pages:
        for r in page.rows:
            if r.kind == "line":
                hist[r.size] += len(r.text)
    return hist.most_common(1)[0][0] if hist else 0.0


def _strip_running_lines(pages: list[_Page]) -> int:
    """Remove headers, footers and page numbers repeated on at least half the pages."""
    if len(pages) < 3:
        return 0

    def key(r: _Row) -> str:
        return re.sub(r"\d+", "#", r.text.lower()).strip()

    def in_zone(r: _Row, height: float) -> bool:
        return r.kind == "line" and (
            r.bottom < height * RUNNING_ZONE or r.top > height * (1 - RUNNING_ZONE)
        )

    seen: Counter[str] = Counter()
    for p in pages:
        seen.update({key(r) for r in p.rows if in_zone(r, p.height)})
    threshold = max(3, len(pages) // 2)
    repeated = {k for k, n in seen.items() if n >= threshold}
    removed = 0
    for p in pages:
        kept = [r for r in p.rows if not (in_zone(r, p.height) and key(r) in repeated)]
        removed += len(p.rows) - len(kept)
        p.rows = kept
    return removed


@dataclass
class _Block:
    kind: str  # "paragraph" | "table" | "bullet"
    lines: list[_Row]

    @property
    def text(self) -> str:
        if self.kind == "table":
            return self.lines[0].text
        out = ""
        for r in self.lines:
            t = r.text
            if not out:
                out = t
            elif re.search(r"[A-Za-z]-$", out) and t[:1].islower():
                out = out[:-1] + t  # re-join a hyphenated word split across lines
            else:
                out += " " + t
        return out


def _blocks(rows: list[_Row]) -> list[_Block]:
    blocks: list[_Block] = []
    prev: _Row | None = None
    for r in rows:
        if r.kind == "table":
            blocks.append(_Block("table", [r]))
        else:
            bullet = bool(_BULLET.match(r.text))
            new = (
                prev is None
                or prev.kind == "table"
                or bullet
                or r.region != prev.region
                or r.top < prev.top
                or r.top - prev.bottom > PARAGRAPH_GAP * r.size
                or abs(r.size - prev.size) > 0.15 * r.size
                or blocks[-1].kind == "bullet"
                and r.x0 < prev.x0 - 1  # left the indented item
            )
            if new:
                blocks.append(_Block("bullet" if bullet else "paragraph", [r]))
            else:
                blocks[-1].lines.append(r)
        prev = r
    return blocks


def _heading_levels(blocks_by_page: list[list[_Block]], body: float) -> dict[float, int]:
    sizes = sorted(
        {
            b.lines[0].size
            for blocks in blocks_by_page
            for b in blocks
            if b.kind == "paragraph" and _looks_like_heading(b, body, large_only=True)
        },
        reverse=True,
    )
    return {s: min(i + 1, 4) for i, s in enumerate(sizes)}


def _looks_like_heading(b: _Block, body: float, *, large_only: bool = False) -> bool:
    first = b.lines[0]
    text = b.text
    if len(b.lines) > 3 or len(text) > MAX_HEADING_CHARS or text.endswith((",", ";")):
        return False
    if body and first.size >= body * HEADING_RATIO:
        return True
    return (
        not large_only
        and first.bold
        and all(r.bold for r in b.lines)
        and len(b.lines) == 1
        and len(text) <= 80
        and not text.endswith(".")
    )


def _clean_title(raw: str | None) -> str | None:
    raw = (raw or "").strip()
    return None if not raw or _JUNK_TITLE.match(raw) else raw


# --- entry point ----------------------------------------------------------------------------


def parse_pdf(data: bytes, *, ocr: OcrEngine | None = None) -> ParsedDocument:
    warnings: list[str] = []
    pages: list[_Page] = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise ParseError(f"The PDF has too many pages (more than {MAX_PAGES:,})")
            if not pdf.pages:
                raise ParseError("The PDF has no pages")
            info = _first_dict(pdf.metadata)
            for number, page in enumerate(pdf.pages, start=1):
                pages.append(_read_page(page, number, ocr, warnings))
                page.flush_cache()
    except ParseError:
        raise
    except Exception as e:  # noqa: BLE001 - library errors on hostile/damaged input vary widely
        log.warning("PDF could not be read (%s)", type(e).__name__, exc_info=True)
        if "password" in f"{type(e).__name__} {e}".lower() or "encrypt" in str(e).lower():
            raise ParseError("The PDF is password protected") from None
        raise ParseError("The PDF could not be read; it may be damaged") from None

    removed = _strip_running_lines(pages)
    body = _body_size(pages)
    blocks_by_page = [_blocks(p.rows) for p in pages]
    levels = _heading_levels(blocks_by_page, body)
    lowest = min(max(levels.values(), default=0) + 1, 4)

    builder = TextBuilder()
    ocr_pages: list[int] = []
    for page, blocks in zip(pages, blocks_by_page, strict=True):
        builder.start_page(page.number)
        if page.ocr_text:
            ocr_pages.append(page.number)
            for para in re.split(r"\n\s*\n", page.ocr_text):
                builder.add(para.replace("\n", " "))
            continue
        previous = ""
        for b in blocks:
            sep_after_bullet = previous == "bullet"
            previous = b.kind
            if b.kind == "table":
                builder.add(b.text)
            elif b.kind == "bullet":
                text = b.text
                m = _BULLET.match(text)
                marker = m.group(1) if m else ""
                rest = text[m.end() :] if m else text
                label = "- " if marker in _SYMBOL_BULLETS else marker + " "
                builder.add(label + rest, sep="\n" if sep_after_bullet else "\n\n")
            elif _looks_like_heading(b, body, large_only=False):
                size = b.lines[0].size
                builder.add(b.text, heading_level=levels.get(size, lowest))
            else:
                builder.add(b.text)
    text, spans, headings = builder.build()
    if not text:
        raise ParseError(
            "The PDF has no extractable text. It looks scanned; install the OCR extra "
            "(Tesseract) or upload a text-based version."
        )
    if ocr_pages:
        warnings.append(f"OCR was used for {len(ocr_pages)} page(s)")
    meta = {
        "parser": "pdf",
        "page_count": len(pages),
        "ocr_pages": ocr_pages,
        "removed_running_lines": removed,
        "body_font_size": body,
    }
    for key, field_name in (
        ("Author", "author"),
        ("Subject", "subject"),
        ("CreationDate", "created"),
    ):
        if info.get(key):
            meta[field_name] = str(info[key]).strip()
    title = _clean_title(info.get("Title")) or next(
        (h.text for h in headings if h.level == 1), None
    )
    return ParsedDocument(
        text=text, title=title, pages=spans, headings=headings, metadata=meta, warnings=warnings
    )


def _first_dict(metadata) -> dict:
    return metadata if isinstance(metadata, dict) else {}
