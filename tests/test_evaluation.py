"""Tests for evaluation metrics."""

from evaluation import estimate_cost, mean_reciprocal_rank, recall_at_k


def test_recall_hit() -> None:
    assert recall_at_k(["a", "b", "c"], ["b"], k=5) == 1


def test_recall_miss() -> None:
    assert recall_at_k(["a", "b", "c"], ["z"], k=5) == 0


def test_recall_cutoff() -> None:
    # relevant doc is outside top-k
    assert recall_at_k(["a", "b", "c", "d", "e", "f"], ["f"], k=3) == 0


def test_mrr_first_position() -> None:
    assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_mrr_second_position() -> None:
    assert mean_reciprocal_rank(["a", "b"], ["b"]) == pytest.approx(0.5)


def test_mrr_no_match() -> None:
    assert mean_reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_estimate_cost_positive() -> None:
    cost = estimate_cost(1000)
    assert cost > 0


import pytest  # noqa: E402 (kept at bottom to avoid circular import issues in test discovery)
