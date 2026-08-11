"""Tests for the capability-claim guards.

The cases are the actual retracted numbers from 2026-08-11, so the guard is
tested against the error it exists to prevent rather than against invented data.
"""
import math

import pytest

from flab import claims

CHANCE = math.log(8)   # 2.0794, the synth tasks' analytic chance level


def test_the_retracted_h2_claim_is_rejected():
    """The real numbers: conflict at 300 steps, both arms at zero accuracy."""
    c = claims.check_forgetting_claim(
        nlls={"ternary": 12.36, "float": 15.80},
        accuracies={"ternary": 0.0, "float": 0.0},
        chance_nll=CHANCE)
    assert not c.ok
    assert any("forgotten completely" in r for r in c.reasons)
    assert any("above chance" in r for r in c.reasons)
    with pytest.raises(AssertionError, match="refusing to report"):
        c.require()


def test_a_real_retention_difference_is_allowed():
    """Disjoint at 300 steps: accuracy well off the floor and NLL near chance,
    so a comparison is meaningful — and it happens to favour the float twin."""
    c = claims.check_forgetting_claim(
        nlls={"ternary": 3.18, "float": 2.30},
        accuracies={"ternary": 0.40, "float": 0.60},
        chance_nll=CHANCE, accuracy_floor=0.125)
    assert c.ok, c.reasons


def test_missing_accuracy_is_rejected_rather_than_assumed_fine():
    c = claims.check_forgetting_claim(
        nlls={"ternary": 2.2, "float": 2.1},
        accuracies={"ternary": None, "float": None},
        chance_nll=CHANCE)
    assert not c.ok
    assert any("no accuracy recorded" in r for r in c.reasons)


def test_one_arm_above_floor_is_enough_to_compare():
    """If either arm retains anything, there is a real difference to discuss."""
    c = claims.check_forgetting_claim(
        nlls={"ternary": 2.5, "float": 1.2},
        accuracies={"ternary": 0.125, "float": 0.55},
        chance_nll=CHANCE, accuracy_floor=0.125)
    assert c.ok, c.reasons


def test_nll_far_above_chance_is_rejected_even_with_good_accuracy():
    """Belt and braces: the headroom rule does not depend on the accuracy rule."""
    c = claims.check_forgetting_claim(
        nlls={"ternary": 9.0, "float": 15.0},
        accuracies={"ternary": 0.5, "float": 0.5},
        chance_nll=CHANCE)
    assert not c.ok
    assert any("softmax tails" in r for r in c.reasons)


# -- matched capability ---------------------------------------------------


def test_the_retracted_matched_capability_claim_is_rejected():
    """0.000532 vs 0.000075 was written up as identical; it is 7x."""
    c = claims.check_matched_capability(
        {"ternary": 0.000532, "float": 0.000075}, label="B-mastery NLL")
    assert not c.ok
    assert "7.1x" in c.reasons[0]


def test_genuinely_matched_values_pass():
    c = claims.check_matched_capability(
        {"ternary": 0.00051, "float": 0.00049}, label="B-mastery NLL")
    assert c.ok


def test_both_exactly_zero_is_matched():
    """At 300 steps both arms hit 0.0000 exactly — that IS matched."""
    c = claims.check_matched_capability(
        {"ternary": 0.0, "float": 0.0}, label="B-mastery NLL")
    assert c.ok
