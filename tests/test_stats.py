"""Tests for aiisco/stats.py, the rank statistics behind the score comparison."""

import pytest

from aiisco.stats import band, mean, ranks, share, spearman, tied_runs, top_band


def test_mean_and_share():
    assert mean([1.0, 2.0, 6.0]) == 3.0
    assert share([True, False, True, True]) == 0.75


def test_tied_runs_groups_equal_values():
    values = [5, 5, 1, 9]
    order = sorted(range(len(values)), key=lambda i: values[i])

    assert list(tied_runs(values, order)) == [(0, 0), (1, 2), (3, 3)]


def test_ranks_are_one_based_and_follow_the_order_of_the_values():
    assert ranks([10, 30, 20]) == [1.0, 3.0, 2.0]


def test_tied_values_share_their_average_rank():
    assert ranks([5, 5, 1]) == [2.5, 2.5, 1.0]
    assert ranks([7, 7, 7]) == [2.0, 2.0, 2.0]


def test_ranks_of_an_empty_list():
    assert ranks([]) == []


def test_spearman_is_one_for_a_perfectly_matching_order():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_is_minus_one_for_a_perfectly_reversed_order():
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_zero_when_one_side_never_varies():
    assert spearman([1, 2, 3], [5, 5, 5]) == 0.0


@pytest.mark.parametrize("score, expected", [
    (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (8, 3), (9, 4), (10, 4),
])
def test_band_maps_the_ten_point_scale_onto_five_rubric_bands(score, expected):
    assert band(score) == expected


def test_band_clamps_scores_outside_the_scale():
    assert band(0) == 0
    assert band(-4) == 0
    assert band(99) == 4


def test_top_band_picks_the_most_likely_band():
    assert top_band([0.1, 0.2, 0.5, 0.15, 0.05]) == 2
    assert top_band([0.5, 0.5, 0.0, 0.0, 0.0]) == 0
