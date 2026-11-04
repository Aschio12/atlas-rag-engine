"""Events of an answer in progress (Phase 4.3), and the small parsers that feed them.

An answer is a stream of events: `retrieval` (what was found, before any text), `token` (a piece
of the answer), `citation` (the first time a source is cited), `done` (the final, validated
answer) and `error`. The same events drive the server-sent stream and the one-shot JSON reply,
which is just the `done` event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.generate.citations import MARKER, labels_in

NOT_FOUND = "NOT_FOUND"
PREFIX_HOLD = len(NOT_FOUND) + 3  # characters to see before deciding the reply is not NOT_FOUND


@dataclass(frozen=True)
class Event:
    name: str  # retrieval | token | citation | done | error
    data: dict[str, Any]

    def sse(self) -> str:
        return f"event: {self.name}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class CitationScanner:
    """Finds citations as the text streams in, even when a marker is split across chunks."""

    def __init__(self, valid: set[str]):
        self._valid = valid
        self._tail = ""
        self.seen: list[str] = []

    def feed(self, text: str) -> list[str]:
        """The valid labels cited for the first time in `text`, in order."""
        data = self._tail + text
        found: list[str] = []
        end = 0
        for match in MARKER.finditer(data):
            for label in labels_in(match.group(1)):
                if label in self._valid and label not in self.seen:
                    self.seen.append(label)
                    found.append(label)
            end = match.end()
        rest = data[end:]
        open_at = rest.rfind("[")
        incomplete = open_at != -1 and "]" not in rest[open_at:] and len(rest) - open_at <= 24
        self._tail = rest[open_at:] if incomplete else ""
        return found


class PrefixGate:
    """Holds back the start of a reply until it is clear whether it is a NOT_FOUND abstention,
    which must not be shown to the user as if it were the answer."""

    def __init__(self) -> None:
        self._held = ""
        self._decided = False
        self.suppressed = False

    def feed(self, text: str) -> str:
        """The text that may be shown now ("" while undecided or suppressed)."""
        if self.suppressed:
            return ""
        if self._decided:
            return text
        self._held += text
        stripped = self._held.lstrip()
        if len(stripped) < PREFIX_HOLD and NOT_FOUND.startswith(stripped.upper()[: len(NOT_FOUND)]):
            return ""  # could still turn out to be NOT_FOUND
        self._decided = True
        if stripped.upper().startswith(NOT_FOUND):
            self.suppressed = True
            return ""
        held, self._held = self._held, ""
        return held

    def flush(self) -> str:
        """Release what is held when the reply ends while still undecided."""
        if self._decided or self.suppressed:
            return ""
        self._decided = True
        if self._held.lstrip().upper().startswith(NOT_FOUND):
            self.suppressed = True
            return ""
        held, self._held = self._held, ""
        return held


def split_answer(text: str, size: int = 24) -> list[str]:
    """Cut a finished answer into token-sized pieces, for when it was held back to be checked."""
    return [text[i : i + size] for i in range(0, len(text), size)]
