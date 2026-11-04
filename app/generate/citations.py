"""Citations (Phase 4.2): checking what the model cites and mapping it back to the documents.

The model cites the context blocks it was given by their ids: `[S1]`, `[S1][S2]`. Short ids are
easy to copy exactly and cheap in tokens, and because the backend assigned them, every id can be
*validated*: a marker that names no block of this answer's context is invented, and is removed
and reported rather than shown as a reference to something that does not exist.

Each valid citation is mapped to its block: document, page, section, the block's character span in
the document and the exact spans of the passages that matched, which is what a UI needs to open
the source at the right place and highlight it.

Two further checks are lexical and deliberately cheap, since they run on every answer:
**support** (what share of a sentence's content words occur in the block it cites) flags a
citation that points at a passage which does not seem to say what is claimed, and **coverage**
(what share of the substantive sentences carry a citation) flags answers that are mostly
unreferenced. Neither proves an answer right; they tell a reader where to look first, and feed the
confidence score.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.ingest.enrichment.language import ALL_STOPWORDS
from app.ingest.sparse import terms
from app.retrieve.types import ContextBlock, Passage

MARKER = re.compile(r"\[\s*([Ss]\d+(?:\s*[,;]\s*[Ss]\d+)*)\s*\]")
LABEL = re.compile(r"[Ss](\d+)")
SUPPORT_MIN = 0.3  # share of a sentence's content words found in the block it cites
SUBSTANTIVE_WORDS = 4  # shorter sentences ("Sure.", "Here is the list:") need no citation

_ATTACHED = re.compile(r"(?<=[.!?])[ \t]+((?:\[S\d+\])+)")
_SENTENCE = re.compile(r"(?:[^\n.!?]|[.!?](?!\s|$|\[S\d))+[.!?]*(?:\[S\d+\])*")
_BULLET = re.compile(r"\s*(?:[-*+\u2022]|\d+[.)])\s+")
_INLINE_GAPS = re.compile(r"(?<=\S)[ \t]{2,}")
_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Citation:
    label: str
    document_id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    page: int | None
    heading_path: list[str]
    char_start: int  # the block's span in the document text
    char_end: int
    chunk_ids: list[uuid.UUID]
    passages: list[Passage]  # the matched chunks inside the block, with their own spans
    mentions: int  # how many sentences cite it
    support: float | None  # best lexical support among those sentences, 0..1
    supported: bool | None  # support >= SUPPORT_MIN (None when it could not be measured)


@dataclass(frozen=True)
class Claim:
    """A sentence of the answer with the sources it cites."""

    text: str
    labels: list[str]
    support: dict[str, float | None]


@dataclass
class Grounding:
    answer: str  # the answer with unusable markers removed and `[S1, S2]` written `[S1][S2]`
    citations: list[Citation] = field(default_factory=list)  # in order of first mention
    claims: list[Claim] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)  # substantive sentences with no citation
    invalid: list[str] = field(default_factory=list)  # labels that name no source, removed
    weak: list[str] = field(default_factory=list)  # cited, but nothing in it matches the claim

    @property
    def coverage(self) -> float | None:
        total = len(self.claims) + len(self.uncited)
        return len(self.claims) / total if total else None

    @property
    def supported_share(self) -> float | None:
        measured = [c for c in self.claims if any(v is not None for v in c.support.values())]
        if not measured:
            return None
        good = sum(1 for c in measured if any((v or 0) >= SUPPORT_MIN for v in c.support.values()))
        return good / len(measured)

    @property
    def score(self) -> float | None:
        """Coverage times supported share: 1.0 = every sentence cited, every citation fits."""
        if self.coverage is None:
            return None
        return self.coverage * (self.supported_share if self.supported_share is not None else 1.0)


def labels_in(marker_text: str) -> list[str]:
    return [f"S{n}" for n in LABEL.findall(marker_text)]


def normalise(answer: str, valid: set[str]) -> tuple[str, list[str]]:
    """Rewrite markers: valid labels as `[S1][S2]`, invalid ones dropped (and returned)."""
    invalid: list[str] = []

    def fix(match: re.Match[str]) -> str:
        keep = []
        for label in labels_in(match.group(1)):
            if label in valid:
                keep.append(label)
            elif label not in invalid:
                invalid.append(label)
        return "".join(f"[{label}]" for label in dict.fromkeys(keep))

    text = MARKER.sub(fix, answer)
    text = _BEFORE_PUNCT.sub(r"\1", text)
    text = _INLINE_GAPS.sub(" ", text)
    return text, invalid


def _content_terms(text: str) -> set[str]:
    return {t for t in terms(text) if t not in ALL_STOPWORDS}


def support_of(sentence: str, block_text: str) -> float | None:
    """Share of the sentence's content words that occur in the block (None if it has none)."""
    wanted = _content_terms(sentence)
    if not wanted:
        return None
    return len(wanted & _content_terms(block_text)) / len(wanted)


def sentences(text: str) -> list[str]:
    """Sentences of an answer, line by line, with markers kept on the sentence they follow. A
    list bullet or number stays with its first sentence."""
    text = _ATTACHED.sub(r"\1", text)
    out: list[str] = []
    for line in text.split("\n"):
        bullet = _BULLET.match(line)
        prefix, body = (bullet.group(0), line[bullet.end() :]) if bullet else ("", line)
        found = [m.group(0).strip() for m in _SENTENCE.finditer(body) if m.group(0).strip()]
        if found:
            found[0] = (prefix.lstrip() + found[0]) if prefix else found[0]
            out.extend(found)
    return out


def _is_substantive(sentence: str) -> bool:
    plain = MARKER.sub("", sentence).strip()
    if plain.startswith("#") or plain.endswith(":"):
        return False
    return len(_WORD.findall(plain)) >= SUBSTANTIVE_WORDS


def ground(answer: str, blocks: list[ContextBlock]) -> Grounding:
    """Validate and map the citations in `answer` against the blocks it was written from."""
    by_label = {b.label: b for b in blocks}
    text, invalid = normalise(answer, set(by_label))
    result = Grounding(answer=text, invalid=invalid)

    best: dict[str, float | None] = {}
    mentions: dict[str, int] = {}
    order: list[str] = []
    for sentence in sentences(text):
        found = list(dict.fromkeys(labels_in(" ".join(MARKER.findall(sentence)))))
        plain = MARKER.sub("", sentence).strip()
        if not found:
            if _is_substantive(sentence):
                result.uncited.append(plain)
            continue
        support = {label: support_of(plain, by_label[label].text) for label in found}
        result.claims.append(Claim(plain, found, support))
        for label in found:
            if label not in mentions:
                order.append(label)
            mentions[label] = mentions.get(label, 0) + 1
            value = support[label]
            if value is not None and (best.get(label) is None or value > (best[label] or 0)):
                best[label] = value
            else:
                best.setdefault(label, value)

    for label in order:
        block = by_label[label]
        support = best.get(label)
        result.citations.append(
            Citation(
                label=label,
                document_id=block.document_id,
                collection_id=block.collection_id,
                title=block.title,
                page=block.page,
                heading_path=list(block.heading_path),
                char_start=block.char_start,
                char_end=block.char_end,
                chunk_ids=list(block.chunk_ids),
                passages=list(block.passages),
                mentions=mentions[label],
                support=support,
                supported=None if support is None else support >= SUPPORT_MIN,
            )
        )
        if support is not None and support < SUPPORT_MIN:
            result.weak.append(label)
    return result
