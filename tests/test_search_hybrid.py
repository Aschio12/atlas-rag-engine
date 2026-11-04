"""Sparse (BM25) and hybrid retrieval through the API."""

import math

import pytest
from search_helpers import PASTA, RECIPES, REFUND, find, ingest, make_collection, search_url


def stage(out, name):
    return next(s for s in out["trace"]["stages"] if s["name"] == name)


# ---- sparse (3.2) ---------------------------------------------------------------------------


async def test_sparse_search_finds_exact_terms_and_explains_the_match(client, indexed):
    out = await find(client, indexed, "Rotterdam warehouse", mode="sparse")
    top = out["hits"][0]
    assert "Rotterdam" in top["text"]
    assert set(top["scores"]) == {"sparse"} and top["scores"]["sparse"] == top["score"]
    assert top["reasons"][0].startswith("sparse: BM25 score")
    assert "matched terms: rotterdam, warehouse" in top["reasons"][0]
    assert [s["name"] for s in out["trace"]["stages"]] == [
        "scope",
        "sparse_search",
        "resolve",
        "result",
    ]  # fmt: skip  (no query embedding: sparse search is purely lexical)


async def test_sparse_search_matches_word_forms_and_ignores_case_and_accents(client, indexed):
    out = await find(client, indexed, "REFUNDED purchases", mode="sparse")
    assert "Refunds are issued" in out["hits"][0]["text"]


async def test_sparse_search_returns_only_passages_sharing_a_term(client, indexed):
    out = await find(client, indexed, "zzzyx qqqqw", mode="sparse", top_k=50)
    assert out["hits"] == []


async def test_a_query_without_usable_terms_finds_nothing_in_sparse_mode(client, indexed):
    out = await find(client, indexed, "? ! a", mode="sparse")
    assert out["hits"] == [] and stage(out, "sparse_search")["details"]["points_found"] == 0


async def test_sparse_search_honours_filters_and_scope(client, app, indexed):
    other = await ingest(client, app, indexed, indexed["url"], RECIPES, "recipes.md")
    only = await find(
        client, indexed, "water", mode="sparse", top_k=50,
        filters=[{"field": "document_id", "op": "eq", "value": other}],
    )  # fmt: skip
    assert only["hits"] and {h["document_id"] for h in only["hits"]} == {other}
    gone = await find(
        client, indexed, "water", mode="sparse", top_k=50,
        filters=[{"field": "language", "op": "eq", "value": "de"}],
    )  # fmt: skip
    assert gone["hits"] == []


async def test_sparse_scores_across_several_collections_are_flagged_as_not_comparable(
    client, app, indexed
):
    cid2, url2 = await make_collection(client, indexed, "cooking")
    await ingest(client, app, indexed, url2, RECIPES, "recipes.md")
    out = await find(client, indexed, "water", mode="sparse")
    assert stage(out, "sparse_search")["details"]["comparable_scores"] is False


# ---- hybrid and fusion (3.3) ----------------------------------------------------------------


async def test_hybrid_is_the_default_and_runs_both_channels(client, indexed):
    r = await client.post(search_url(indexed), json={"query": REFUND}, headers=indexed["headers"])
    out = r.json()
    assert r.status_code == 200 and out["mode"] == "hybrid"
    names = [s["name"] for s in out["trace"]["stages"]]
    assert names == [
        "scope", "embed_query", "dense_search", "sparse_search", "resolve", "fusion", "result",
    ]  # fmt: skip
    top = out["hits"][0]
    assert "Refunds are issued" in top["text"]
    assert set(top["scores"]) == {"dense", "sparse", "rrf"} and top["scores"]["rrf"] == top["score"]
    assert math.isclose(top["score"], 1 / 61 + 1 / 61)  # first in both lists, k=60
    kinds = [r.split(":")[0] for r in top["reasons"]]
    assert kinds == ["dense", "sparse", "reciprocal rank fusion"]
    assert "dense #1, sparse #1" in top["reasons"][2]
    assert stage(out, "fusion")["details"]["found_by_several_channels"] >= 1


async def test_fusion_orders_hits_by_fused_score_and_ranks_them(client, indexed):
    out = await find(client, indexed, "refund warranty hardware", mode="hybrid", top_k=10)
    scores = [h["score"] for h in out["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert [h["rank"] for h in out["hits"]] == list(range(1, len(scores) + 1))
    assert len({h["chunk_id"] for h in out["hits"]}) == len(scores)


async def test_a_passage_found_by_only_one_channel_still_appears(client, indexed):
    # "limited warranty" shares terms with one passage; dense returns everything, sparse a few
    out = await find(client, indexed, "limited warranty", mode="hybrid", top_k=50)
    only_dense = [h for h in out["hits"] if "sparse" not in h["scores"]]
    both = [h for h in out["hits"] if "sparse" in h["scores"]]
    assert both and only_dense  # sparse found a few, dense filled in the rest
    assert all(h["scores"]["rrf"] > 0 for h in out["hits"])
    assert max(h["score"] for h in both) >= max(h["score"] for h in only_dense)


async def test_weighted_fusion_normalises_and_weights_the_channels(client, indexed):
    out = await find(
        client, indexed, REFUND, mode="hybrid",
        fusion={"method": "weighted", "dense_weight": 1, "sparse_weight": 3},
    )  # fmt: skip
    top = out["hits"][0]
    assert set(top["scores"]) == {"dense", "sparse", "weighted"}
    assert math.isclose(top["score"], 1 * 1.0 + 3 * 1.0)  # best in both lists: 1.0 normalised
    assert "weighted fusion" in top["reasons"][-1]
    details = stage(out, "fusion")["details"]
    assert details["method"] == "weighted" and details["k"] is None
    assert details["weights"] == {"dense": 1.0, "sparse": 3.0}


async def test_a_zero_weight_channel_does_not_change_the_order_of_the_other(client, indexed):
    dense_only = await find(client, indexed, "refund policy shipping", mode="dense", top_k=10)
    hybrid = await find(
        client, indexed, "refund policy shipping", mode="hybrid", top_k=10,
        fusion={"sparse_weight": 0},
    )  # fmt: skip
    assert [h["chunk_id"] for h in hybrid["hits"]] == [h["chunk_id"] for h in dense_only["hits"]]


async def test_rrf_k_is_reported_and_changes_scores(client, indexed):
    small = await find(client, indexed, REFUND, mode="hybrid", fusion={"k": 1})
    large = await find(client, indexed, REFUND, mode="hybrid", fusion={"k": 1000})
    assert math.isclose(small["hits"][0]["score"], 2 / 2)
    assert math.isclose(large["hits"][0]["score"], 2 / 1001)
    assert stage(small, "fusion")["details"]["k"] == 1


async def test_pool_limits_candidates_per_channel(client, indexed):
    out = await find(client, indexed, "customer care team", mode="hybrid", pool=10, top_k=50)
    sizes = stage(out, "fusion")["details"]["list_sizes"]
    assert sizes["dense"] <= 10 and sizes["sparse"] <= 10


@pytest.mark.parametrize(
    "body",
    [
        {"mode": "hybrid", "min_score": 0.5},
        {"mode": "sparse", "min_score": 0.5},
        {"fusion": {"dense_weight": 0, "sparse_weight": 0}},
        {"fusion": {"k": 0}},
        {"fusion": {"method": "borda"}},
        {"fusion": {"dense_weight": -1}},
        {"pool": 9},
        {"pool": 101},
    ],
)
async def test_invalid_retrieval_options_are_422(client, indexed, body):
    r = await client.post(
        search_url(indexed), json={"query": "x", **body}, headers=indexed["headers"]
    )
    assert r.status_code == 422


async def test_min_score_still_works_in_dense_mode(client, indexed):
    out = await find(client, indexed, REFUND, mode="dense", min_score=0.99, top_k=50)
    assert all(h["score"] >= 0.99 for h in out["hits"])


async def test_hybrid_works_across_collections_with_different_models(client, app, space):
    cid1, url1 = await make_collection(client, space, "a")
    cid2, url2 = await make_collection(client, space, "b", embed_model="other-model")
    from search_helpers import POLICY

    await ingest(client, app, space, url1, POLICY, "policy.md")
    await ingest(client, app, space, url2, RECIPES, "recipes.md")
    out = await find(client, space, PASTA, mode="hybrid", top_k=20)
    assert {h["collection_id"] for h in out["hits"]} == {cid1, cid2}
    assert "spaghetti" in out["hits"][0]["text"]
