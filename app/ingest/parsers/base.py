"""What every parser produces: one normalised text plus a map of its structure.

All offsets (`Heading.char_start`, `PageSpan.char_*`, and later every chunk's `char_start` and
`char_end`) are indexes into `ParsedDocument.text`, measured in Python characters (code points).
That single coordinate system is what lets a citation highlight the exact passage in the viewer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

PARSED_VERSION = 1


class ParseError(Exception):
    """The file cannot be parsed and retrying will not help. The message is shown to the user."""


@dataclass(frozen=True)
class Heading:
    level: int  # 1 (top) .. 6
    text: str
    char_start: int
    page: int | None = None  # page or slide number, when the format has pages


@dataclass(frozen=True)
class PageSpan:
    page: int  # 1-based page or slide number
    char_start: int
    char_end: int


@dataclass
class ParsedDocument:
    text: str
    title: str | None = None
    pages: list[PageSpan] = field(default_factory=list)
    headings: list[Heading] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def page_at(self, offset: int) -> int | None:
        """The page containing a character offset, or None for formats without pages."""
        for span in self.pages:
            if span.char_start <= offset < span.char_end:
                return span.page
        return None

    def summary(self) -> dict[str, Any]:
        """A compact description stored on the document row (no text)."""
        return {
            "title": self.title,
            "characters": len(self.text),
            "pages": len(self.pages) or None,
            "headings": len(self.headings),
            "warnings": self.warnings,
            **self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PARSED_VERSION,
            "text": self.text,
            "title": self.title,
            "pages": [[p.page, p.char_start, p.char_end] for p in self.pages],
            "headings": [[h.level, h.text, h.char_start, h.page] for h in self.headings],
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParsedDocument:
        return cls(
            text=d["text"],
            title=d.get("title"),
            pages=[PageSpan(*p) for p in d.get("pages", [])],
            headings=[Heading(*h) for h in d.get("headings", [])],
            metadata=d.get("metadata", {}),
            warnings=d.get("warnings", []),
        )


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​‎‏  ﻿]")
_SPACES = re.compile(r"[ \t  -   　]+")
_BLANK_RUNS = re.compile(r"\n{3,}")


def normalize(text: str, *, keep_indent: bool = False) -> str:
    """Canonical text: NFC, `\\n` newlines, no control or zero-width characters, collapsed
    runs of spaces, no trailing spaces and at most one blank line in a row.

    With `keep_indent` leading whitespace is preserved (code blocks, preformatted text).
    """
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    out = []
    for line in text.split("\n"):
        indent = ""
        if keep_indent:
            indent = line[: len(line) - len(line.lstrip(" \t"))].replace("\t", "    ")
            line = line.lstrip(" \t")
        out.append(indent + _SPACES.sub(" ", line).strip(" "))
    return _BLANK_RUNS.sub("\n\n", "\n".join(out)).strip("\n")


class TextBuilder:
    """Assembles `text` block by block while recording headings and page spans, so the offsets
    are correct by construction rather than searched for afterwards."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.headings: list[Heading] = []
        self.pages: list[PageSpan] = []
        self._page: int | None = None
        self._page_start = 0

    @property
    def length(self) -> int:
        return self._length

    def start_page(self, page: int) -> None:
        self._close_page()
        self._page = page
        self._page_start = self._length

    def _close_page(self) -> None:
        if self._page is not None:
            self.pages.append(PageSpan(self._page, self._page_start, self._length))
            self._page = None

    def add(
        self, text: str, *, sep: str = "\n\n", heading_level: int | None = None, keep_indent=False
    ) -> int | None:
        """Append a block. Returns its start offset, or None if it was empty."""
        text = normalize(text, keep_indent=keep_indent)
        if not text:
            return None
        if self._parts:
            self._parts.append(sep)
            self._length += len(sep)
            if self._page is not None and self._page_start == self._length - len(sep):
                # The separator belongs between pages, not inside the new page's span.
                self._page_start = self._length
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        if heading_level is not None:
            self.headings.append(
                Heading(min(max(heading_level, 1), 6), text.replace("\n", " "), start, self._page)
            )
        return start

    def build(self) -> tuple[str, list[PageSpan], list[Heading]]:
        self._close_page()
        return "".join(self._parts), self.pages, self.headings
