"""How retrieved passages are shown to the model (Phase 4.1).

Each context block becomes a `<source>` element with an id the model cites (`[S1]`) and the
metadata it needs to cite sensibly (title, page, section). Everything inside is *data*: the
system prompt says so, and this module backs it up by making sure a passage cannot close the
element early or open a fake one, which is the usual way a retrieved document tries to pass off
its own text as instructions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from app.retrieve.types import ContextBlock

_TAG = re.compile(r"<\s*(/?)\s*(sources?)\b", re.IGNORECASE)


def escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def neutralise(text: str) -> str:
    """Defuse anything in a passage that looks like our own <source> or <sources> tags."""
    return _TAG.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


def render_sources(
    blocks: list[ContextBlock],
    *,
    texts: Mapping[str, str] | None = None,
    flagged: Mapping[str, str] | None = None,
) -> str:
    """The <sources> element for a prompt.

    `texts` replaces a block's text by label (used for redacted text); `flagged` marks sources
    that look like they contain instructions, with the reason, so the model is told which ones
    to be extra careful with.
    """
    parts = ["<sources>"]
    for block in blocks:
        attrs = [f'id="{escape_attr(block.label)}"', f'title="{escape_attr(block.title)}"']
        if block.page is not None:
            attrs.append(f'page="{block.page}"')
        if block.heading_path:
            attrs.append(f'section="{escape_attr(" > ".join(block.heading_path))}"')
        if flagged and block.label in flagged:
            attrs.append(f'warning="{escape_attr(flagged[block.label])}"')
        text = (texts or {}).get(block.label, block.text)
        parts.append(f"<source {' '.join(attrs)}>\n{neutralise(text)}\n</source>")
    parts.append("</sources>")
    return "\n".join(parts)


def user_message(sources: str, question: str) -> str:
    return f"{sources}\n\nQuestion: {question}"
