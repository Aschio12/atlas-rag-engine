"""Retrieval quality metrics (Phase 3.9).

Each function takes `relevance`: for one query, whether each ranked result was relevant, best
rank first (`[False, True, False]` means the second result was the first relevant one), plus
`total_relevant`, the number of relevant things that exist for the query.

* **recall@k**: the share of the relevant things found in the top k.
* **hit@k**: 1 if anything relevant is in the top k, else 0 (averaged: "how often does the
  answer show up at all").
* **MRR**: 1 / rank of the first relevant result, 0 if none (averaged over queries). It rewards
  putting a right answer first.
* **nDCG@k**: relevant results count more the higher they rank (gain / log2(rank + 1)),
  normalised by the best possible ordering, so 1.0 is perfect. Gains are binary here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        raise ValueError("a query with nothing relevant has no recall")
    return min(sum(relevance[:k]), total_relevant) / total_relevant


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: Sequence[bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        raise ValueError("a query with nothing relevant has no nDCG")
    dcg = sum(1.0 / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1) if rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(total_relevant, k) + 1))
    return dcg / ideal


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(q / 100 * len(ordered)) - 1, 0)
    return ordered[index]
