import math

import pytest

from app.retrieve.fusion import reciprocal_rank_fusion, weighted_fusion


def keys(fused):
    return [f.key for f in fused]


def test_rrf_scores_follow_the_formula_for_a_known_ranking():
    out = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "sparse": ["b", "a", "d"]}, k=60)
    by_key = {f.key: f for f in out}
    assert math.isclose(by_key["a"].score, 1 / 61 + 1 / 62)
    assert math.isclose(by_key["b"].score, 1 / 62 + 1 / 61)
    assert math.isclose(by_key["c"].score, 1 / 63)
    assert math.isclose(by_key["d"].score, 1 / 63)
    assert by_key["a"].ranks == {"dense": 1, "sparse": 2}
    assert by_key["a"].contributions == {"dense": 1 / 61, "sparse": 1 / 62}


def test_rrf_rewards_agreement_over_a_single_first_place():
    # "x" is first in one list only; "y" is second in both, and wins
    out = reciprocal_rank_fusion({"one": ["x", "y"], "two": ["z", "y"]})
    assert keys(out)[0] == "y"


def test_rrf_ties_break_by_best_rank_then_key():
    out = reciprocal_rank_fusion({"one": ["b", "a"], "two": ["a", "b"]})
    assert keys(out) == ["a", "b"]  # equal scores and equal best rank: key order
    out = reciprocal_rank_fusion({"one": ["q"], "two": ["p"]})
    assert keys(out) == ["p", "q"]


def test_rrf_k_changes_how_much_the_head_matters():
    rankings = {"one": ["a", "b", "c", "d"], "two": ["d", "c", "b", "a"]}
    small = {f.key: f.score for f in reciprocal_rank_fusion(rankings, k=1)}
    large = {f.key: f.score for f in reciprocal_rank_fusion(rankings, k=1000)}
    assert small["a"] / small["b"] > large["a"] / large["b"] or small["a"] == small["d"]
    assert max(large.values()) - min(large.values()) < max(small.values()) - min(small.values())


def test_rrf_weights_scale_each_list():
    out = reciprocal_rank_fusion(
        {"dense": ["a", "b"], "sparse": ["b", "a"]}, weights={"dense": 2.0, "sparse": 1.0}
    )
    assert keys(out) == ["a", "b"]
    assert math.isclose(out[0].score, 2 / 61 + 1 / 62)
    zero = reciprocal_rank_fusion({"dense": ["a"], "sparse": ["b"]}, weights={"sparse": 0.0})
    assert keys(zero) == ["a", "b"] and zero[1].score == 0.0


def test_rrf_counts_a_repeated_key_once_per_list():
    out = reciprocal_rank_fusion({"one": ["a", "a", "b"]})
    assert out[0].key == "a" and math.isclose(out[0].score, 1 / 61)
    assert out[1].ranks == {"one": 3}  # positions are not renumbered


def test_rrf_with_nothing_to_fuse():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"one": []}) == []


def test_rrf_single_list_keeps_its_order():
    assert keys(reciprocal_rank_fusion({"one": ["c", "a", "b"]})) == ["c", "a", "b"]


@pytest.mark.parametrize("bad", [0, -3])
def test_rrf_rejects_a_nonsense_k(bad):
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"one": ["a"]}, k=bad)


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"one": ["a"]}, weights={"one": -1})
    with pytest.raises(ValueError):
        weighted_fusion({"one": {"a": 1.0}}, weights={"one": -1})


def test_weighted_fusion_normalises_scales_before_adding():
    # BM25-like scores in the tens, cosine in 0..1: raw addition would let sparse win by scale
    out = weighted_fusion(
        {"dense": {"a": 0.9, "b": 0.5, "c": 0.1}, "sparse": {"b": 30.0, "c": 20.0, "a": 10.0}}
    )
    scores = {f.key: f.score for f in out}
    assert math.isclose(scores["a"], 1.0 + 0.0)
    assert math.isclose(scores["b"], 0.5 + 1.0)
    assert math.isclose(scores["c"], 0.0 + 0.5)
    assert keys(out) == ["b", "a", "c"]


def test_weighted_fusion_applies_weights_and_treats_missing_as_zero():
    out = weighted_fusion(
        {"dense": {"a": 1.0, "b": 0.0}, "sparse": {"c": 5.0, "a": 1.0}},
        weights={"dense": 0.25, "sparse": 0.75},
    )
    scores = {f.key: f.score for f in out}
    assert math.isclose(scores["a"], 0.25 * 1.0 + 0.75 * 0.0)
    assert math.isclose(scores["b"], 0.0)
    assert math.isclose(scores["c"], 0.75 * 1.0)
    assert keys(out)[0] == "c"
    assert {f.key: dict(f.ranks) for f in out}["c"] == {"sparse": 1}


def test_weighted_fusion_with_a_flat_list_treats_items_as_equal():
    out = weighted_fusion({"one": {"a": 3.0, "b": 3.0}})
    assert [f.score for f in out] == [1.0, 1.0]


def test_weighted_fusion_with_nothing_to_fuse():
    assert weighted_fusion({}) == [] and weighted_fusion({"one": {}}) == []


def test_fusion_is_deterministic_regardless_of_list_order():
    a = reciprocal_rank_fusion({"x": ["a", "b"], "y": ["b", "a"]})
    b = reciprocal_rank_fusion({"y": ["b", "a"], "x": ["a", "b"]})
    assert keys(a) == keys(b)
