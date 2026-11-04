import math

import pytest

from app.retrieve.metrics import (
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_relevant_results_in_the_top_k_over_all_relevant():
    rel = [False, True, True, False, True]
    assert recall_at_k(rel, 3, 1) == 0.0
    assert recall_at_k(rel, 3, 3) == pytest.approx(2 / 3)
    assert recall_at_k(rel, 3, 5) == 1.0
    assert recall_at_k(rel, 4, 5) == 0.75  # one relevant thing was never retrieved
    assert recall_at_k([True, True], 1, 2) == 1.0  # never above 1


def test_hit_rate():
    assert hit_at_k([False, False, True], 2) == 0.0
    assert hit_at_k([False, False, True], 3) == 1.0
    assert hit_at_k([], 5) == 0.0


def test_reciprocal_rank_rewards_the_first_relevant_result():
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False, True, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_ndcg_is_one_for_a_perfect_ordering_and_lower_when_relevant_results_sink():
    assert ndcg_at_k([True, True, False], 2, 3) == pytest.approx(1.0)
    perfect = ndcg_at_k([True, False, False], 1, 3)
    sunk = ndcg_at_k([False, False, True], 1, 3)
    assert perfect == 1.0 and sunk == pytest.approx(1 / math.log2(4)) and sunk < perfect


def test_ndcg_matches_a_hand_calculation():
    rel = [True, False, True]  # relevant at ranks 1 and 3; two things are relevant
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(rel, 2, 3) == pytest.approx(dcg / ideal)


def test_ndcg_ideal_is_capped_by_k():
    # three things are relevant but only the top 2 count: the ideal is two perfect results
    assert ndcg_at_k([True, True, True], 3, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("fn", [recall_at_k, ndcg_at_k])
def test_a_query_with_nothing_relevant_is_an_error(fn):
    with pytest.raises(ValueError):
        fn([True], 0, 1)


def test_mean_and_percentile():
    assert mean([1, 2, 3]) == 2 and mean([]) == 0.0
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 50) == 50 and percentile(values, 95) == 100
    assert percentile(values, 0) == 10 and percentile([], 95) == 0.0
    assert percentile([5], 95) == 5
