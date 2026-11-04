"""Confidence and abstention (Phase 4.5): knowing when not to answer.

A grounded assistant that guesses when its documents are silent is worse than none. Confidence is
computed from signals that exist before and after the model writes anything:

* **retrieval**: how relevant the best passage looks. The reranker's score is the best signal
  (a cross-encoder reads question and passage together); without one, the dense cosine
  similarity of the best hit; lexical-only searches have no semantic signal at all.
* **coverage**: the share of the question's content words found in the retrieved context. A
  question about "battery warranty on the X200" whose sources never mention an X200 scores low
  however similar the sources otherwise look.
* **grounding** (after generation): how many of the answer's sentences cite a source whose text
  fits the claim (see `citations`).

Before generation the score decides whether to call the model at all: below `abstain_below` the
answer is "I couldn't find this" and no tokens are spent. After generation the score is the badge
shown with the answer. The model can also abstain itself by replying NOT_FOUND.

The numbers are only as meaningful as the models behind them: reranker and embedding scores are
not calibrated across providers, so treat the defaults as a starting point and tune
`abstain_below` against questions of your own (the Phase 6 evaluation harness is for that).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.generate.citations import Grounding
from app.ingest.enrichment.language import ALL_STOPWORDS
from app.ingest.sparse import terms
from app.retrieve.types import ContextBlock, Hit

Level = Literal["high", "medium", "low", "none"]
HIGH, MEDIUM, LOW = 0.7, 0.45, 0.2
DEFAULT_ABSTAIN_BELOW = 0.15
SEMANTIC_WEIGHT = 0.65  # retrieval vs coverage when there is a semantic signal
LEXICAL_CAP = 0.8  # with no semantic signal, coverage alone cannot make us fully sure


@dataclass
class Confidence:
    score: float
    level: Level
    signals: dict[str, float | None] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def abstain(self) -> bool:
        return self.level == "none"


def level_of(score: float, abstain_below: float = DEFAULT_ABSTAIN_BELOW) -> Level:
    if score < abstain_below:
        return "none"
    if score >= HIGH:
        return "high"
    if score >= MEDIUM:
        return "medium"
    return "low"


def content_terms(text: str) -> set[str]:
    return {t for t in terms(text) if t not in ALL_STOPWORDS and not re.fullmatch(r"\d", t)}


def coverage(question: str, blocks: list[ContextBlock]) -> float | None:
    """Share of the question's content words that occur anywhere in the context."""
    wanted = content_terms(question)
    if not wanted:
        return None
    found: set[str] = set()
    for block in blocks:
        found |= wanted & content_terms(block.text)
        if found == wanted:
            break
    return len(found) / len(wanted)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def retrieval_signal(hits: list[Hit]) -> tuple[float | None, str]:
    """The best semantic relevance signal available among the hits, and where it came from."""
    rerank = [h.scores["rerank"] for h in hits if "rerank" in h.scores]
    if rerank:
        return _clamp(max(rerank)), "reranker"
    dense = [s for h in hits for k, s in h.scores.items() if k == "dense" or k.startswith("dense[")]
    if dense:
        return _clamp(max(dense)), "dense similarity"
    return None, ""


def assess(
    question: str,
    hits: list[Hit],
    blocks: list[ContextBlock],
    *,
    abstain_below: float = DEFAULT_ABSTAIN_BELOW,
) -> Confidence:
    """Confidence in the retrieved context, before any answer exists."""
    if not blocks:
        return Confidence(0.0, "none", {}, ["no passage was retrieved"])
    semantic, source = retrieval_signal(hits)
    cover = coverage(question, blocks)
    signals = {"retrieval": semantic, "coverage": cover}
    reasons: list[str] = []
    if semantic is not None:
        reasons.append(f"best passage scores {semantic:.2f} ({source})")
    if cover is not None:
        reasons.append(f"{cover:.0%} of the question's key words appear in the sources")
    if semantic is not None and cover is not None:
        score = SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * cover
    elif semantic is not None:
        score = semantic
    elif cover is not None:
        score = cover * LEXICAL_CAP
        reasons.append("no semantic score is available (keyword search only)")
    else:
        score = 0.5  # nothing measurable in the question (e.g. a single stopword): stay neutral
        reasons.append("the question has no key words to check against the sources")
    score = _clamp(score)
    return Confidence(score, level_of(score, abstain_below), signals, reasons)


def finalize(
    before: Confidence, grounding: Grounding | None, *, abstain_below: float = DEFAULT_ABSTAIN_BELOW
) -> Confidence:
    """Blend in how well the written answer is grounded: the badge shown with the answer."""
    if grounding is None or grounding.score is None:
        return before
    score = _clamp(0.5 * before.score + 0.5 * grounding.score)
    reasons = list(before.reasons)
    cited = len(grounding.claims)
    total = cited + len(grounding.uncited)
    reasons.append(f"{cited} of {total} sentences cite a source")
    if grounding.weak:
        reasons.append(
            f"{len(grounding.weak)} citation(s) point at text that does not match the claim"
        )
    if grounding.invalid:
        reasons.append(f"{len(grounding.invalid)} citation(s) named sources that do not exist")
        score *= 0.9
    signals = {**before.signals, "grounding": grounding.score}
    return Confidence(score, level_of(score, abstain_below), signals, reasons)
