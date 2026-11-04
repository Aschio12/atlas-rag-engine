"""Reranking stage (Phase 3.4): re-score the fused candidates with a cross-encoder.

Retrieval casts a wide net cheaply (embeddings, BM25); a reranker then reads the query together
with each candidate and orders them far more accurately, but costs a model pass per pair. So the
pipeline reranks only the best `candidates` (default 50) and keeps the top `top_k` (say 8).

A (query, passage) score does not change while the passage and the model stay the same, so scores
are cached by (reranker, normalised query, chunk id, passage hash) in a small in-process LRU with
a time limit. A repeated question, or a follow-up that retrieves mostly the same passages, only
pays for the pairs it has not seen. Editing a passage changes its hash, so a stale score is never
reused for new text.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from app.providers.base import Reranker
from app.retrieve.types import Hit

MAX_TEXT_CHARS = 2000


class RerankUnavailable(Exception):
    """Reranking was requested but no reranker is configured."""


def normalise_query(query: str) -> str:
    return " ".join(query.lower().split())


def rerank_text(hit: Hit) -> str:
    """What the reranker reads: the title and section (they often carry the topic) and the
    passage."""
    head = " > ".join([hit.title, *hit.heading_path])
    return f"{head}\n{hit.text}"[:MAX_TEXT_CHARS]


class RerankCache:
    """An LRU of (query, passage) scores with a time-to-live. `max_entries=0` turns it off."""

    def __init__(
        self,
        max_entries: int = 20_000,
        ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._max = max_entries
        self._ttl = ttl_s
        self._clock = clock
        self._data: OrderedDict[tuple[str, str, str, str], tuple[float, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    @staticmethod
    def key(reranker: Reranker, query: str, hit: Hit) -> tuple[str, str, str, str]:
        text_hash = hashlib.sha1(rerank_text(hit).encode("utf-8")).hexdigest()
        return (
            f"{reranker.name}:{reranker.model}",
            normalise_query(query),
            str(hit.chunk_id),
            text_hash,
        )

    def get(self, key: tuple[str, str, str, str]) -> float | None:
        entry = self._data.get(key)
        if entry is None or self._clock() - entry[1] > self._ttl:
            if entry is not None:
                del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key: tuple[str, str, str, str], score: float) -> None:
        if self._max <= 0:
            return
        self._data[key] = (score, self._clock())
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


@dataclass
class RerankOutcome:
    hits: list[Hit]  # reordered, best first; ranks are renumbered
    details: dict = field(default_factory=dict)


async def rerank_hits(
    reranker: Reranker, cache: RerankCache, query: str, hits: list[Hit]
) -> RerankOutcome:
    """Order `hits` by the reranker's score. Cached pairs are not sent again."""
    if not hits:
        return RerankOutcome([], {"candidates": 0, "scored": 0, "cached": 0})
    keys = [cache.key(reranker, query, h) for h in hits]
    scores: dict[int, float] = {}
    for i, key in enumerate(keys):
        cached = cache.get(key)
        if cached is not None:
            scores[i] = cached
    fresh = [i for i in range(len(hits)) if i not in scores]
    cached_count = len(scores)
    if fresh:
        result = await reranker.rerank(query, [rerank_text(hits[i]) for i in fresh], len(fresh))
        for r in result:
            if 0 <= r.index < len(fresh):
                scores[fresh[r.index]] = r.score
                cache.put(keys[fresh[r.index]], r.score)
    unscored = [i for i in range(len(hits)) if i not in scores]
    # a candidate the reranker did not score is ranked below every scored one, in its old order
    order = sorted(scores, key=lambda i: (-scores[i], i)) + unscored

    before = [h.rank for h in hits]  # by position in `hits`, captured before ranks change
    out: list[Hit] = []
    movements: list[dict] = []
    for new_rank, i in enumerate(order, start=1):
        hit = hits[i]
        if i in scores:
            hit.scores["rerank"] = scores[i]
            hit.score = scores[i]
            shift = before[i] - new_rank
            move = "stayed" if shift == 0 else f"moved {'up' if shift > 0 else 'down'} {abs(shift)}"
            note = "" if i in fresh else " (cached)"
            hit.reasons.append(
                f"rerank ({reranker.model}): {scores[i]:.4f}{note}, {move} "
                f"(#{before[i]} -> #{new_rank})"
            )
        else:
            hit.reasons.append(f"rerank: {reranker.model} did not score this; kept below the rest")
        hit.rank = new_rank
        out.append(hit)
        movements.append(
            {
                "chunk_id": str(hit.chunk_id),
                "from": before[i],
                "to": new_rank,
                "score": scores.get(i),
                "cached": i not in fresh,
            }
        )
    details = {
        "reranker": f"{reranker.name}:{reranker.model}",
        "candidates": len(hits),
        "cached": cached_count,
        "scored": len(fresh) - len(unscored),
        "unscored": len(unscored),
        "movements": movements,
    }
    return RerankOutcome(out, details)
