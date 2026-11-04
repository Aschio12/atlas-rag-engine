"""Fusing several ranked lists into one (Phase 3.3).

Hybrid retrieval produces one ranking per channel (dense, sparse, and one per query variant).
Their scores live on different scales (cosine similarity, BM25), so they cannot simply be added.

* **Reciprocal Rank Fusion** uses only ranks: an item scores `sum(weight / (k + rank))` over the
  lists it appears in. It needs no calibration, is hard to game with one outlier score and is
  the default. `k` (60 in the original paper) flattens the head of each list: a larger `k` makes
  agreement between lists matter more than a first place in one of them.
* **Weighted fusion** min-max normalises each list's scores to 0..1 and sums them with weights.
  It keeps score magnitudes, so a runaway-confident channel can dominate; use it when you have
  tuned the weights against your own questions.

Both are pure functions over item keys, so they are easy to test with known rankings. Ties break
by the best rank an item had anywhere, then by key, so results are deterministic.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fused:
    key: Hashable
    score: float
    ranks: Mapping[str, int]  # 1-based rank in each list the item appeared in
    contributions: Mapping[str, float] = field(default_factory=dict)  # what each list added


def _order(items: dict[Hashable, tuple[float, dict[str, int], dict[str, float]]]) -> list[Fused]:
    fused = [Fused(k, s, ranks, contrib) for k, (s, ranks, contrib) in items.items()]
    fused.sort(key=lambda f: (-f.score, min(f.ranks.values()), str(f.key)))
    return fused


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Hashable]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[Fused]:
    """Fuse ranked lists of keys (best first). A key repeated within one list counts once, at
    its first position."""
    if k < 1:
        raise ValueError("k must be at least 1")
    weights = weights or {}
    acc: dict[Hashable, tuple[float, dict[str, int], dict[str, float]]] = {}
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {name!r} must not be negative")
        seen: set[Hashable] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            part = weight / (k + rank)
            score, ranks, contrib = acc.setdefault(key, (0.0, {}, {}))
            ranks[name] = rank
            contrib[name] = part
            acc[key] = (score + part, ranks, contrib)
    return _order(acc)


def _normalise(scores: Mapping[Hashable, float]) -> dict[Hashable, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:  # a list that cannot tell its items apart: all equally good
        return {k: 1.0 for k in scores}
    return {k: (v - low) / (high - low) for k, v in scores.items()}


def weighted_fusion(
    scored: Mapping[str, Mapping[Hashable, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> list[Fused]:
    """Fuse lists of `key -> score` after min-max normalising each list. An item missing from a
    list gets 0 from it."""
    weights = weights or {}
    acc: dict[Hashable, tuple[float, dict[str, int], dict[str, float]]] = {}
    for name, scores in scored.items():
        weight = weights.get(name, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {name!r} must not be negative")
        normal = _normalise(scores)
        order = sorted(scores, key=lambda k: (-scores[k], str(k)))
        for rank, key in enumerate(order, start=1):
            part = weight * normal[key]
            score, ranks, contrib = acc.setdefault(key, (0.0, {}, {}))
            ranks[name] = rank
            contrib[name] = part
            acc[key] = (score + part, ranks, contrib)
    return _order(acc)
