"""PowerPoint decks (.pptx): one page per slide, in presentation order.

Each slide contributes its title (as a level-1 heading), its text in reading order (top to
bottom, left to right), tables, and the speaker notes, which often hold the real explanation.
"""

from __future__ import annotations

from xml.etree.ElementTree import Element

from app.ingest.parsers.base import ParsedDocument, ParseError, TextBuilder
from app.ingest.parsers.ooxml import core_properties, open_package, q, read_xml, relationships

TITLE_TYPES = {"title", "ctrTitle"}
MAX_SLIDES = 2_000


def _paragraphs(txbody: Element) -> list[str]:
    out = []
    for p in txbody.findall(q("a:p")):
        parts = []
        for el in p.iter():
            if el.tag == q("a:t"):
                parts.append(el.text or "")
            elif el.tag == q("a:br"):
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            out.append(line)
    return out


def _placeholder_type(shape: Element) -> str | None:
    ph = shape.find(f"{q('p:nvSpPr')}/{q('p:nvPr')}/{q('p:ph')}")
    if ph is None:
        return None
    return ph.get("type") or "body"  # a placeholder without a type is a body placeholder


def _position(shape: Element) -> tuple[int, int]:
    off = shape.find(f"{q('p:spPr')}/{q('a:xfrm')}/{q('a:off')}")
    if off is None:
        return (10**9, 10**9)  # inherited from the layout: keep document order
    try:
        return int(off.get("y", 0)), int(off.get("x", 0))
    except ValueError:
        return (10**9, 10**9)


def _table_rows(tbl: Element) -> list[str]:
    rows = []
    for tr in tbl.findall(q("a:tr")):
        cells = []
        for tc in tr.findall(q("a:tc")):
            body = tc.find(q("a:txBody"))
            text = " ".join(_paragraphs(body)) if body is not None else ""
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _notes_text(zf, slide_path: str) -> str:
    for target, kind in relationships(zf, slide_path).values():
        if kind == "notesSlide":
            root = read_xml(zf, target, required=False)
            if root is None:
                return ""
            lines: list[str] = []
            for sp in root.iter(q("p:sp")):
                if _placeholder_type(sp) == "body":
                    body = sp.find(q("p:txBody"))
                    if body is not None:
                        lines.extend(_paragraphs(body))
            return " ".join(lines)
    return ""


def _slide_paths(zf) -> list[str]:
    presentation = read_xml(zf, "ppt/presentation.xml")
    rels = relationships(zf, "ppt/presentation.xml")
    paths = []
    for sld in presentation.iter(q("p:sldId")):
        target = rels.get(sld.get(q("r:id"), ""))
        if target and target[1] == "slide":
            paths.append(target[0])
    return paths


def parse_pptx(data: bytes) -> ParsedDocument:
    zf = open_package(data, "PowerPoint")
    slide_paths = _slide_paths(zf)
    if not slide_paths:
        raise ParseError("The presentation has no slides")
    if len(slide_paths) > MAX_SLIDES:
        raise ParseError(f"The presentation has too many slides (more than {MAX_SLIDES:,})")

    builder = TextBuilder()
    with_notes = 0
    for number, path in enumerate(slide_paths, start=1):
        builder.start_page(number)
        slide = read_xml(zf, path)
        title_blocks: list[str] = []
        items: list[tuple[tuple[int, int, int], str]] = []  # (reading-order key, text)
        for order, el in enumerate(slide.iter()):
            if el.tag == q("p:sp"):
                body = el.find(q("p:txBody"))
                if body is None:
                    continue
                lines = _paragraphs(body)
                if not lines:
                    continue
                kind = _placeholder_type(el)
                if kind in TITLE_TYPES:
                    title_blocks.append(" ".join(lines))
                elif kind not in {"sldNum", "dt", "ftr"}:  # slide number, date, footer
                    items.append(((*_position(el), order), "\n".join(lines)))
            elif el.tag == q("a:tbl"):
                rows = _table_rows(el)
                if rows:
                    items.append(((10**9, 10**9, order), "\n".join(rows)))
        if title_blocks:
            builder.add(" ".join(title_blocks), heading_level=1)
        for _, text in sorted(items, key=lambda i: i[0]):
            builder.add(text, sep="\n\n")
        notes = _notes_text(zf, path)
        if notes:
            with_notes += 1
            builder.add(f"Speaker notes: {notes}")
    text, pages, headings = builder.build()
    if not text:
        raise ParseError("The presentation contains no text")
    props = core_properties(zf)
    title = props.get("title") or (headings[0].text if headings else None)
    metadata = {
        "parser": "pptx",
        "slides": len(slide_paths),
        "slides_with_notes": with_notes,
        **{k: v for k, v in props.items() if k != "title"},
    }
    return ParsedDocument(text=text, title=title, pages=pages, headings=headings, metadata=metadata)
