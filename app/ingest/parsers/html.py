"""HTML to structured text using only the standard library.

Boilerplate (scripts, styles, navigation, footers, forms) is dropped, headings become headings,
list items get bullets, table rows become `cell | cell` lines and `<pre>` keeps its layout.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from app.ingest.files import decode_text
from app.ingest.parsers.base import ParsedDocument, ParseError, TextBuilder, normalize

SKIP = {
    "script", "style", "noscript", "template", "svg", "head", "iframe", "nav", "footer",
    "aside", "form", "button", "select", "option", "textarea", "canvas", "object", "embed",
}  # fmt: skip
VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "source", "wbr"}
BLOCKS = {
    "p", "div", "section", "article", "main", "header", "blockquote", "figure", "figcaption",
    "ul", "ol", "dl", "dt", "dd", "table", "thead", "tbody", "tfoot", "details", "summary",
    "address", "fieldset", "legend", "caption",
}  # fmt: skip
HEADINGS = {f"h{i}": i for i in range(1, 7)}
_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.IGNORECASE)


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = TextBuilder()
        self.title: str | None = None
        self.lang: str | None = None
        self.meta: dict[str, str] = {}
        self._skip: list[str] = []
        self._buf: list[str] = []
        self._heading: int | None = None
        self._pre = 0
        self._in_title = False
        self._title_buf: list[str] = []
        self._li = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._last: str | None = None  # kind of the previous block: item, row or None

    # -- helpers --------------------------------------------------------------------------
    def _flush(self) -> None:
        raw = "".join(self._buf)
        self._buf = []
        if self._pre:
            text = normalize(raw, keep_indent=True)
        else:
            text = normalize(re.sub(r"\s+", " ", raw))
        if not text:
            return
        if self._heading is not None:
            self.builder.add(text, heading_level=self._heading)
            self._last = None
        elif self._li:
            self._add_grouped("- " + text, "item")
        else:
            self.builder.add(text, keep_indent=bool(self._pre))
            self._last = None

    def _add_grouped(self, text: str, kind: str) -> None:
        """List items and table rows sit on consecutive lines; other blocks get a blank line."""
        self.builder.add(text, sep="\n" if self._last == kind else "\n\n")
        self._last = kind

    # -- parser callbacks -----------------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html" and a.get("lang"):
            self.lang = a["lang"]
        if tag == "meta":
            name = (a.get("name") or a.get("property") or "").lower()
            if name in {"description", "author", "og:title"} and a.get("content"):
                self.meta[name] = a["content"].strip()
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in SKIP:
            if tag not in VOID:
                self._skip.append(tag)
            return
        if self._skip:
            if tag == self._skip[-1] and tag not in VOID:
                self._skip.append(tag)  # nested element of the same kind: keep the count right
            return
        if "hidden" in a or (a.get("aria-hidden") or "").lower() == "true":
            self._skip.append(tag)
            return
        if tag in VOID:
            if tag == "br":
                self._buf.append("\n" if self._pre else " ")
            elif tag == "hr":
                self._flush()
            return
        if tag in HEADINGS:
            self._flush()
            self._heading = HEADINGS[tag]
        elif tag == "pre":
            self._flush()
            self._pre += 1
        elif tag == "li":
            self._flush()
            self._li += 1
        elif tag == "tr":
            self._flush()
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag in BLOCKS:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self._in_title = False
            self.title = normalize(" ".join("".join(self._title_buf).split())) or None
            return
        if self._skip:
            if tag == self._skip[-1]:
                self._skip.pop()
            return
        if tag in HEADINGS and self._heading is not None:
            self._flush()
            self._heading = None
        elif tag == "pre":
            self._flush()
            self._pre = max(0, self._pre - 1)
        elif tag == "li":
            self._flush()
            self._li = max(0, self._li - 1)
        elif tag in ("td", "th") and self._cell is not None:
            if self._row is not None:
                cell = normalize(" ".join("".join(self._cell).split()))
                if cell:
                    self._row.append(cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._add_grouped(" | ".join(self._row), "row")
            self._row = None
        elif tag in BLOCKS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self._title_buf.append(data)
            return
        if self._skip:
            return
        if self._cell is not None:
            self._cell.append(data)
        else:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def _decode(data: bytes) -> tuple[str, str]:
    """Honour a <meta charset> declaration when the bytes are not valid UTF-8."""
    text, encoding = decode_text(data)
    if encoding == "cp1252" and (m := _CHARSET.search(data[:4096])):
        try:
            return data.decode(m.group(1).decode("ascii")), m.group(1).decode("ascii").lower()
        except (LookupError, UnicodeDecodeError):
            pass
    return text, encoding


def parse_html(data: bytes) -> ParsedDocument:
    raw, encoding = _decode(data)
    extractor = _Extractor()
    extractor.feed(raw)
    extractor.close()
    text, _, headings = extractor.builder.build()
    if not text:
        raise ParseError("The page has no readable text")
    title = extractor.title or extractor.meta.get("og:title")
    if not title:
        title = next((h.text for h in headings if h.level == 1), None)
    metadata = {"parser": "html", "encoding": encoding}
    if extractor.lang:
        metadata["language"] = extractor.lang
    for key in ("description", "author"):
        if key in extractor.meta:
            metadata[key] = extractor.meta[key]
    return ParsedDocument(text=text, title=title, headings=headings, metadata=metadata)
