import math

import pytest

from recsys.metrics import candidate_recall, ranking_metrics


def test_hand_computed_example():
    recs = {1: [10, 11, 12], 2: [20, 21, 22]}
    truth = {1: {11, 99}, 2: {30}}
    m = ranking_metrics(recs, truth, k=3, n_items=100)
    # user 1: one hit of two at rank 2; user 2: no hits
    ndcg_1 = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert m["recall"] == pytest.approx((0.5 + 0.0) / 2)
    assert m["ndcg"] == pytest.approx(ndcg_1 / 2)
    assert m["hit"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx((1 / 3) / 2)
    assert m["coverage"] == pytest.approx(6 / 100)
    assert m["users"] == 2


def test_perfect_ranking_has_ndcg_one():
    m = ranking_metrics({1: [5, 6, 7, 8]}, {1: {5, 6}}, k=4, n_items=10)
    assert m["ndcg"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_users_without_truth_are_skipped_and_empty_is_none():
    assert ranking_metrics({}, {}, k=10, n_items=10)["recall"] is None
    assert ranking_metrics({1: [1]}, {1: set()}, k=10, n_items=10)["users"] == 0


def test_candidate_recall():
    assert candidate_recall({1: {1, 2, 3}, 2: {9}}, {1: {1, 4}, 2: {9}}) == pytest.approx(0.75)
