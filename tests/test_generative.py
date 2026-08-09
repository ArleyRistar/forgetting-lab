"""Tests for generative exact-match normalisation (open item 14).

These pin the normaliser against 2606.27634's behaviour, including where that
behaviour is imperfect — reproducing it is the point, improving it would not be
a replication.
"""
import pytest

from flab import generative as g


@pytest.mark.parametrize("raw,want", [
    ("B", "B"),
    ("B.", "B"),
    ("The answer is C", "C"),
    ("Final answer: D", "D"),
    ("option A", "A"),
    ("**B**", "B"),                       # markdown stripped
    ("<think>hmm</think>C", "C"),         # thinking block stripped
    ("B<|eot_id|>", "B"),                 # chat terminator stripped
    ("I think it is B because ...", "B"),
])
def test_multiple_choice_extraction(raw, want):
    assert g.normalize_answer(raw, "FOMC") == want


@pytest.mark.parametrize("raw,want", [
    ("42", "42"),
    ("42.0", "42"),                       # integer-valued floats canonicalise
    ("1,234", "1234"),                    # commas removed
    ("The answer is 17.", "17"),
    ("Therefore, 8", "8"),
    ("step 1: 5 apples, step 2: so 12", "12"),   # last number wins
])
def test_numeric_extraction(raw, want):
    assert g.normalize_answer(raw, "NumGLUE-cm") == want


def test_boxed_is_caught_only_incidentally():
    """Their normaliser has NO \\boxed{} handling despite the NumGLUE prompt
    asking for it — the last-number fallback catches it by accident. Pinned
    because reproducing their behaviour is the point; a normaliser that handled
    boxes properly would score differently and would not be a replication."""
    assert g.normalize_answer(r"so the answer is \boxed{42}", "NumGLUE-cm") == "42"
    # ...and it breaks when anything numeric follows the box, which is exactly
    # the fragility we are inheriting on purpose.
    assert g.normalize_answer(r"\boxed{42} (see step 3)", "NumGLUE-cm") == "3"


def test_gold_and_prediction_normalise_the_same_way():
    """Both sides go through the same function — otherwise a formatting
    difference in the gold answer would count as a wrong prediction."""
    assert g.normalize_answer("B", "FOMC") == g.normalize_answer("Answer: B", "FOMC")
    assert g.normalize_answer("186", "NumGLUE-cm") == g.normalize_answer("186.0", "NumGLUE-cm")
