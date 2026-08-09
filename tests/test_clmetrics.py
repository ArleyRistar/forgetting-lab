"""Tests for the continual-learning metrics (phase-1b task 4).

Pure arithmetic, no model — these guard the definitions and especially the sign
convention, which is where a plausible-looking but backwards number would come
from.
"""
import math

import pytest

from flab import clmetrics


# -- sign convention -----------------------------------------------------


def test_forgetting_is_negative_bwt_under_both_observables():
    """The whole point of the convention: one direction means forgetting,
    whichever observable you read."""
    # Task 0 is learned then decays; task 1 is learned last and holds.
    acc = [[0.9, 0.2],
           [0.5, 0.9]]          # accuracy: task 0 fell 0.9 -> 0.5
    nll = [[0.1, 2.0],
           [0.6, 0.1]]          # NLL: task 0 rose 0.1 -> 0.6 (same event)
    m_acc = clmetrics.compute(acc, [0.3, 0.3], "accuracy")
    m_nll = clmetrics.compute(nll, [2.0, 2.0], "nll")
    assert m_acc.bwt < 0 and m_nll.bwt < 0
    assert m_acc.bwt == pytest.approx(-0.4)
    assert m_nll.bwt == pytest.approx(-0.5)   # -(0.6) - -(0.1)


def test_perfect_retention_gives_zero_bwt():
    acc = [[0.9, 0.2],
           [0.9, 0.9]]
    assert clmetrics.compute(acc, [0.3, 0.3], "accuracy").bwt == pytest.approx(0.0)


def test_backward_improvement_gives_positive_bwt():
    acc = [[0.5, 0.2],
           [0.8, 0.9]]          # task 0 improved after task 1 was learned
    assert clmetrics.compute(acc, [0.3, 0.3], "accuracy").bwt == pytest.approx(0.3)


# -- definitions ---------------------------------------------------------


def test_acc_is_the_mean_of_the_final_boundary():
    acc = [[0.9, 0.2], [0.5, 0.9]]
    assert clmetrics.compute(acc, [0.3, 0.3], "accuracy").acc == pytest.approx(0.7)


def test_fwt_compares_untrained_tasks_against_the_baseline():
    """Task 1 before it is trained (0.2) vs the untouched model (0.3):
    training on task 0 made it worse, so FWT is negative."""
    acc = [[0.9, 0.2], [0.5, 0.9]]
    assert clmetrics.compute(acc, [0.3, 0.3], "accuracy").fwt == pytest.approx(-0.1)


def test_nll_negation_matches_the_shakedown_direction():
    """The real 1a numbers: ScienceQA was never trained on and rose from
    1.6353 to 1.8220 — worse — so FWT must come out negative."""
    nll = [[1.06, 1.94, 1.75],
           [1.45, 0.86, 1.82],
           [1.76, 0.97, 0.77]]
    base = [5.151, 1.6355, 1.6353]
    m = clmetrics.compute(nll, base, "nll")
    assert m.bwt < 0, "FOMC and Py150 both degraded by the end"
    assert m.fwt < 0, "training on earlier tasks hurt the later ones"


def test_three_task_matrix_shape_is_validated():
    with pytest.raises(ValueError):
        clmetrics.compute([[0.1, 0.2]], [0.1, 0.2], "accuracy")     # too few boundaries
    with pytest.raises(ValueError):
        clmetrics.compute([[0.1], [0.2]], [0.1, 0.2], "accuracy")   # ragged row


def test_unknown_observable_rejected():
    with pytest.raises(ValueError, match="observable"):
        clmetrics.compute([[0.1]], [0.1], "perplexity")


def test_single_task_has_undefined_transfer_not_a_fake_zero():
    m = clmetrics.compute([[0.5]], [0.3], "accuracy")
    assert m.acc == pytest.approx(0.5)
    assert math.isnan(m.bwt) and math.isnan(m.fwt)


# -- correlation ---------------------------------------------------------


def test_pearson_recovers_a_known_negative_relationship():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    r, p, n = clmetrics.pearson(xs, ys)
    assert r == pytest.approx(-1.0)
    assert n == 5 and p is not None and p < 0.01


def test_pearson_on_noisy_negative_data_keeps_the_sign():
    xs = [0.1, 0.3, 0.5, 0.8, 1.2, 1.6]
    ys = [0.62, 0.60, 0.55, 0.51, 0.44, 0.33]
    r, p, _ = clmetrics.pearson(xs, ys)
    assert -1.0 <= r < -0.8
    assert p is not None and p < 0.05


def test_pearson_refuses_to_invent_a_correlation_from_two_points():
    r, p, n = clmetrics.pearson([1.0, 2.0], [2.0, 1.0])
    assert math.isnan(r) and p is None and n == 2


def test_pearson_handles_a_constant_series():
    r, p, _ = clmetrics.pearson([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert math.isnan(r)
