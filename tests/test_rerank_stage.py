import copy
import uuid

import pytest
from search_helpers import REFUND, find, search_url

from app.providers.base import Reranked, Reranker
from app.providers.errors import ProviderError
from app.retrieve.rerank import RerankCache, normalise_query, rerank_hits, rerank_text
from app.retrieve.types import Hit


class Scripted(Reranker):
    """Scores each document with a function of its text and records what it was asked."""

    name = "scripted"
    model = "m1"

    def __init__(self, score=lambda text: 0.0, drop=()):
        self.score, self.drop = score, set(drop)
        self.calls: list[list[str]] = []

    async def rerank(self, query, documents, top_n):
        self.calls.append(list(documents))
        out = [Reranked(i, self.score(d)) for i, d in enumerate(documents) if i not in self.drop]
        out.sort(key=lambda r: (-r.score, r.index))
        return out[:top_n]


def hit(text, rank, title="T", heading=("H",)):
    return Hit(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), collection_id=uuid.uuid4(),
        rank=rank, ordinal=0,
        score=1.0 / rank, text=text, title=title, page=None, char_start=0, char_end=len(text),
        heading_path=list(heading), parent_id=None, language="en", source_uri=None,
        matched="chunk", scores={"rrf": 1.0 / rank}, reasons=["fused"],
    )  # fmt: skip


def texts(hits):
    return [h.text for h in hits]


# ---- cache ----------------------------------------------------------------------------------


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_cache_returns_what_was_stored_and_counts_hits_and_misses():
    cache, key = RerankCache(), ("r", "q", "c", "h")
    assert cache.get(key) is None
    cache.put(key, 0.7)
    assert cache.get(key) == 0.7 and (cache.hits, cache.misses) == (1, 1)


def test_cache_entries_expire_after_the_ttl():
    clock = Clock()
    cache, key = RerankCache(ttl_s=10, clock=clock), ("r", "q", "c", "h")
    cache.put(key, 0.5)
    clock.now = 9.9
    assert cache.get(key) == 0.5
    clock.now = 10.1
    assert cache.get(key) is None and len(cache) == 0


def test_cache_evicts_the_least_recently_used_entry():
    cache = RerankCache(max_entries=2)
    a, b, c = ("r", "q", "a", "h"), ("r", "q", "b", "h"), ("r", "q", "c", "h")
    cache.put(a, 1)
    cache.put(b, 2)
    cache.get(a)  # a is now fresher than b
    cache.put(c, 3)
    assert cache.get(b) is None and cache.get(a) == 1 and cache.get(c) == 3


def test_a_cache_of_size_zero_stores_nothing():
    cache = RerankCache(max_entries=0)
    cache.put(("r", "q", "c", "h"), 1)
    assert len(cache) == 0


def test_keys_ignore_query_case_and_spacing_but_not_model_or_passage_text():
    h, r = hit("same text", 1), Scripted()
    assert RerankCache.key(r, "Refund  Policy", h) == RerankCache.key(r, "refund policy", h)
    assert normalise_query("  A   b ") == "a b"
    other_model = Scripted()
    other_model.model = "m2"
    assert RerankCache.key(r, "q", h) != RerankCache.key(other_model, "q", h)
    h.text = "edited text"  # same chunk id, new content: the old score must not apply
    assert RerankCache.key(r, "q", h) != RerankCache.key(r, "q", hit("same text", 1))


def test_the_reranker_reads_title_section_and_passage():
    assert rerank_text(hit("body", 1, title="Policy", heading=("Terms", "Refunds"))) == (
        "Policy > Terms > Refunds\nbody"
    )


# ---- stage ----------------------------------------------------------------------------------


async def test_hits_are_reordered_by_score_and_ranks_renumbered():
    hits = [hit("low", 1), hit("high", 2), hit("mid", 3)]
    scores = {"low": 0.1, "high": 0.9, "mid": 0.5}
    out = await rerank_hits(Scripted(lambda d: scores[d.split("\n")[1]]), RerankCache(), "q", hits)
    assert texts(out.hits) == ["high", "mid", "low"]
    assert [h.rank for h in out.hits] == [1, 2, 3]
    assert [h.score for h in out.hits] == [0.9, 0.5, 0.1]
    top = out.hits[0]
    assert top.scores == {"rrf": 0.5, "rerank": 0.9}
    assert top.reasons[-1] == "rerank (m1): 0.9000, moved up 1 (#2 -> #1)"
    assert "moved up 1" in out.hits[1].reasons[-1]
    assert "moved down 2" in out.hits[2].reasons[-1]
    assert out.details["movements"][0] == {
        "chunk_id": str(top.chunk_id), "from": 2, "to": 1, "score": 0.9, "cached": False,
    }  # fmt: skip
    assert (out.details["candidates"], out.details["scored"], out.details["cached"]) == (3, 3, 0)


async def test_a_repeated_query_is_served_from_the_cache():
    originals, cache = [hit("a", 1), hit("b", 2)], RerankCache()
    reranker = Scripted(lambda d: 1.0 if d.endswith("b") else 0.0)
    first = await rerank_hits(reranker, cache, "refunds", copy.deepcopy(originals))
    again = await rerank_hits(reranker, cache, "  Refunds ", copy.deepcopy(originals))
    assert len(reranker.calls) == 1  # the second call sent nothing
    assert again.details["cached"] == 2 and again.details["scored"] == 0
    assert all(m["cached"] for m in again.details["movements"])
    assert "(cached)" in again.hits[0].reasons[-1]
    assert [h.scores["rerank"] for h in first.hits] == [1.0, 0.0]


async def test_only_unseen_pairs_are_sent_to_the_reranker():
    cache = RerankCache()
    seen = [hit("a", 1), hit("b", 2)]
    reranker = Scripted(lambda d: 0.5)
    await rerank_hits(reranker, cache, "q", copy.deepcopy(seen))
    # a follow-up retrieves "b" again plus a new passage; `seen` keeps their chunk ids
    follow_up = [hit("c", 1), copy.deepcopy(seen[1])]
    out = await rerank_hits(reranker, cache, "q", follow_up)
    assert len(reranker.calls[1]) == 1 and reranker.calls[1][0].endswith("c")
    assert out.details["cached"] == 1 and out.details["scored"] == 1


async def test_a_different_query_does_not_reuse_scores():
    cache, hits = RerankCache(), [hit("a", 1)]
    reranker = Scripted()
    await rerank_hits(reranker, cache, "first question", hits)
    await rerank_hits(reranker, cache, "second question", hits)
    assert len(reranker.calls) == 2


async def test_passages_the_reranker_skips_rank_below_scored_ones():
    hits = [hit("a", 1), hit("b", 2), hit("c", 3)]
    out = await rerank_hits(Scripted(lambda d: 0.4, drop={0}), RerankCache(), "q", hits)
    assert texts(out.hits) == ["b", "c", "a"]
    assert out.details["unscored"] == 1 and out.details["scored"] == 2
    assert "did not score" in out.hits[-1].reasons[-1]
    assert "rerank" not in out.hits[-1].scores


async def test_nothing_to_rerank_makes_no_call():
    reranker = Scripted()
    out = await rerank_hits(reranker, RerankCache(), "q", [])
    assert out.hits == [] and reranker.calls == []


async def test_ties_keep_the_incoming_order():
    hits = [hit("a", 1), hit("b", 2), hit("c", 3)]
    out = await rerank_hits(Scripted(lambda d: 0.5), RerankCache(), "q", hits)
    assert texts(out.hits) == ["a", "b", "c"]


# ---- in the search API ------------------------------------------------------------------------


def use(app, reranker):
    app.state.providers._reranker = reranker
    app.state.providers._reranker_built = True
    return reranker


class Reverse(Reranker):
    """Prefers whatever the first stage liked least."""

    name = "reverse"
    model = "rev"

    async def rerank(self, query, documents, top_n):
        out = [Reranked(i, float(i)) for i in range(len(documents))]
        out.sort(key=lambda r: -r.score)
        return out[:top_n]


async def test_asking_for_reranking_without_a_reranker_is_a_clear_400(client, indexed):
    r = await client.post(
        search_url(indexed),
        json={"query": "x", "rerank": {"enabled": True}},
        headers=indexed["headers"],
    )
    assert r.status_code == 400 and "RERANK_PROVIDER" in r.json()["detail"]


async def test_reranking_reorders_the_fused_hits_and_explains_it(client, app, indexed):
    use(app, Reverse())
    plain = await find(client, indexed, REFUND, mode="hybrid", top_k=4)
    reranked = await find(
        client, indexed, REFUND, mode="hybrid", top_k=4, rerank={"enabled": True, "candidates": 4}
    )
    assert [h["chunk_id"] for h in reranked["hits"]] == [h["chunk_id"] for h in plain["hits"]][::-1]
    top = reranked["hits"][0]
    assert top["scores"]["rerank"] == top["score"] and "rrf" in top["scores"]
    assert top["reasons"][-1].startswith("rerank (rev):") and "moved up" in top["reasons"][-1]
    assert [h["rank"] for h in reranked["hits"]] == [1, 2, 3, 4]
    stage = next(s for s in reranked["trace"]["stages"] if s["name"] == "rerank")
    assert stage["details"]["reranker"] == "reverse:rev" and stage["details"]["candidates"] == 4
    assert reranked["warnings"] == []


async def test_the_reranker_reads_the_candidate_window_and_top_k_is_cut_after(client, app, indexed):
    reranker = use(app, Scripted(lambda d: 0.5))
    out = await find(
        client, indexed, "customer care", mode="hybrid", top_k=2,
        rerank={"enabled": True, "candidates": 6},
    )  # fmt: skip
    assert len(out["hits"]) == 2 and len(reranker.calls[0]) == 6
    small = await find(
        client, indexed, "team", mode="hybrid", top_k=5, rerank={"enabled": True, "candidates": 1}
    )
    assert len(small["hits"]) == 5  # the window is never smaller than top_k


async def test_repeating_a_search_reuses_cached_scores(client, app, indexed):
    reranker = use(app, Scripted(lambda d: 0.5))
    opts = {"mode": "hybrid", "top_k": 3, "rerank": {"enabled": True, "candidates": 5}}
    first = await find(client, indexed, "warranty", **opts)
    second = await find(client, indexed, "warranty", **opts)
    assert len(reranker.calls) == 1
    stage = next(s for s in second["trace"]["stages"] if s["name"] == "rerank")
    assert stage["details"]["cached"] == 5 and stage["details"]["scored"] == 0
    assert [h["chunk_id"] for h in first["hits"]] == [h["chunk_id"] for h in second["hits"]]


async def test_a_failing_reranker_degrades_to_the_fused_order_with_a_warning(client, app, indexed):
    class Down(Reranker):
        name, model = "down", "x"

        async def rerank(self, query, documents, top_n):
            raise ProviderError("down", "HTTP 503")

    use(app, Down())
    plain = await find(client, indexed, REFUND, mode="hybrid", top_k=3)
    out = await find(client, indexed, REFUND, mode="hybrid", top_k=3, rerank={"enabled": True})
    assert [h["chunk_id"] for h in out["hits"]] == [h["chunk_id"] for h in plain["hits"]]
    assert "Reranking failed" in out["warnings"][0] and "503" in out["warnings"][0]
    stage = next(s for s in out["trace"]["stages"] if s["name"] == "rerank")
    assert stage["details"]["failed"] is True
    assert all("rerank" not in h["scores"] for h in out["hits"])


@pytest.mark.parametrize("body", [{"candidates": 0}, {"candidates": 101}, {"enabled": "maybe"}])
async def test_invalid_rerank_options_are_422(client, indexed, body):
    r = await client.post(
        search_url(indexed), json={"query": "x", "rerank": body}, headers=indexed["headers"]
    )
    assert r.status_code == 422


async def test_reranking_works_in_every_mode(client, app, indexed):
    use(app, Reverse())
    for mode in ("dense", "sparse", "hybrid"):
        out = await find(
            client, indexed, "refund warranty", mode=mode, top_k=3, rerank={"enabled": True}
        )
        assert out["hits"] and all("rerank" in h["scores"] for h in out["hits"]), mode
