"""Word documents (.docx): headings, paragraphs, bullets and tables, in document order."""

from __future__ import annotations

import re
from xml.etree.ElementTree import Element

from app.ingest.parsers.base import ParsedDocument, ParseError, TextBuilder
from app.ingest.parsers.ooxml import core_properties, open_package, q, read_xml

_HEADING_NAME = re.compile(r"^heading\s*([1-9])$", re.IGNORECASE)
_LIST_STYLE = re.compile(r"list\s*(bullet|number|paragraph|continue)", re.IGNORECASE)
MAX_STYLE_DEPTH = 6


class _Styles:
    """styleId -> (name, basedOn), used to recognise heading and list styles, including styles
    derived from them ("My Heading" based on "Heading 1")."""

    def __init__(self, root: Element | None):
        self._by_id: dict[str, tuple[str, str | None]] = {}
        if root is None:
            return
        for style in root.findall(q("w:style")):
            sid = style.get(q("w:styleId"))
            if not sid:
                continue
            name_el = style.find(q("w:name"))
            based = style.find(q("w:basedOn"))
            self._by_id[sid] = (
                (name_el.get(q("w:val")) if name_el is not None else "") or sid,
                based.get(q("w:val")) if based is not None else None,
            )

    def _chain(self, style_id: str | None):
        seen = 0
        while style_id and seen < MAX_STYLE_DEPTH:
            name, based = self._by_id.get(style_id, (style_id, None))
            yield name
            style_id, seen = based, seen + 1

    def heading_level(self, style_id: str | None) -> int | None:
        for name in self._chain(style_id):
            if m := _HEADING_NAME.match(name):
                return int(m.group(1))
            if name.lower() == "title":
                return 1
            if name.lower() == "subtitle":
                return 2
        return None

    def is_list(self, style_id: str | None) -> bool:
        return any(_LIST_STYLE.search(name) for name in self._chain(style_id))


def _paragraph_text(p: Element) -> str:
    out: list[str] = []
    for el in p.iter():
        if el.tag == q("w:t"):
            out.append(el.text or "")
        elif el.tag == q("w:tab"):
            out.append("\t")
        elif el.tag in (q("w:br"), q("w:cr")):
            if el.get(q("w:type")) != "page":
                out.append("\n")
        elif el.tag == q("w:noBreakHyphen"):
            out.append("-")
    return "".join(out)


def _style_id(p: Element) -> str | None:
    el = p.find(f"{q('w:pPr')}/{q('w:pStyle')}")
    return el.get(q("w:val")) if el is not None else None


def _outline_level(p: Element) -> int | None:
    el = p.find(f"{q('w:pPr')}/{q('w:outlineLvl')}")
    if el is None:
        return None
    try:
        level = int(el.get(q("w:val"), ""))
    except ValueError:
        return None
    return level + 1 if 0 <= level <= 8 else None


class _Walker:
    def __init__(self, styles: _Styles):
        self.styles = styles
        self.builder = TextBuilder()
        self._last: str | None = None

    def paragraph(self, p: Element) -> None:
        text = _paragraph_text(p)
        if not text.strip():
            return
        style = _style_id(p)
        level = self.styles.heading_level(style) or _outline_level(p)
        if level is not None and len(text.strip()) <= 300:
            self.builder.add(text, heading_level=level)
            self._last = None
        elif p.find(f"{q('w:pPr')}/{q('w:numPr')}") is not None or self.styles.is_list(style):
            self.builder.add("- " + text.strip(), sep="\n" if self._last == "item" else "\n\n")
            self._last = "item"
        else:
            self.builder.add(text)
            self._last = None

    def table(self, tbl: Element) -> None:
        self._last = None  # a table always starts a new block
        for tr in tbl.findall(q("w:tr")):
            cells = []
            for tc in tr.findall(q("w:tc")):
                parts = [_paragraph_text(p).strip() for p in tc.iter(q("w:p"))]
                cell = " ".join(x for x in parts if x)
                if cell:
                    cells.append(cell)
            if cells:
                sep = "\n" if self._last == "row" else "\n\n"
                self.builder.add(" | ".join(cells), sep=sep)
                self._last = "row"

    def walk(self, parent: Element) -> None:
        for child in parent:
            if child.tag == q("w:p"):
                self.paragraph(child)
            elif child.tag == q("w:tbl"):
                self.table(child)
            elif child.tag in (q("w:sdt"), q("w:customXml"), q("w:smartTag")):
                content = child.find(q("w:sdtContent"))
                self.walk(content if content is not None else child)


def parse_docx(data: bytes) -> ParsedDocument:
    zf = open_package(data, "Word")
    document = read_xml(zf, "word/document.xml")
    body = document.find(q("w:body")) if document is not None else None
    if body is None:
        raise ParseError("The Word document has no body")
    walker = _Walker(_Styles(read_xml(zf, "word/styles.xml", required=False)))
    walker.walk(body)
    text, _, headings = walker.builder.build()
    if not text:
        raise ParseError("The Word document contains no text")
    props = core_properties(zf)
    title = props.get("title") or next((h.text for h in headings if h.level == 1), None)
    metadata = {"parser": "docx", **{k: v for k, v in props.items() if k != "title"}}
    return ParsedDocument(text=text, title=title, headings=headings, metadata=metadata)
