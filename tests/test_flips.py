"""Tests for weight-state flip instrumentation (phase 1d, task 1).

Hand-built tensors throughout, not a model: the phase-1b KL direction bug
survived a model-based test because a small model's output distribution is
near-uniform and hides almost everything.

Each test targets a *distinct* way the instrument can be silently wrong. The
partition tests matter most — an earlier design defined the third term as a
residual, which made its consistency check true by construction.
"""
import pytest
import torch

from flab import bitlinear as bl
from flab import flips


# -- the state must be the forward pass's state ---------------------------


def test_state_matches_the_quantiser_used_in_the_forward_pass():
    torch.manual_seed(0)
    w = torch.randn(256, 128)
    st = flips.state(w, flips.tensor_scale(w))
    assert torch.equal(st, torch.sign(bl.weight_quant(w)))


def test_state_matches_the_quantiser_at_exact_ties():
    """`torch.round` is round-half-to-even. An implementation using `>= 0.5`
    would disagree here and silently drift from what the model computes."""
    # mean|w| = 1 exactly, so scaled values are the raw ones and 0.5/1.5 are ties.
    w = torch.tensor([[0.5, -0.5, 1.5, -1.5, 1.0, -1.0]])
    assert torch.isclose(w.abs().mean(), torch.tensor(1.0))
    st = flips.state(w, flips.tensor_scale(w))
    assert torch.equal(st, torch.sign(bl.weight_quant(w)))


def test_state_takes_only_three_values():
    torch.manual_seed(0)
    w = torch.randn(64, 64) * 17.0
    st = flips.state(w, flips.tensor_scale(w))
    assert set(torch.unique(st).tolist()) <= {-1.0, 0.0, 1.0}


# -- the partition --------------------------------------------------------


def test_no_change_means_no_flips():
    """The null any broken implementation fails."""
    torch.manual_seed(0)
    w = torch.randn(128, 64)
    s = flips.flip_partition(w, w.clone())
    assert s.flipped == 0 and s.cancelled == 0
    s.check()


def test_pure_scale_motion_is_classified_scale_only():
    """Scaling every weight by a constant leaves the *states* untouched only if
    the threshold scales too — it does, since the scale is 1/mean|w|. So force a
    genuine boundary move by changing the distribution's spread while holding
    the weights that matter fixed."""
    # Two groups: a fixed probe weight near the boundary, and bulk weights whose
    # magnitude we change to move mean|w| and hence the threshold.
    w_prev = torch.tensor([[0.6, 1.0, 1.0, 1.0]])
    w_cur = torch.tensor([[0.6, 2.0, 2.0, 2.0]])   # same probe, bigger mean
    s = flips.flip_partition(w_prev, w_cur)
    s.check()
    # the probe weight flipped purely because the boundary moved under it
    assert s.scale_only >= 1
    assert s.weight_only == 0


def test_pure_weight_motion_with_the_scale_pinned_is_weight_only():
    """Move one weight across the boundary while keeping mean|w| identical, so
    the scale cannot be responsible."""
    # mean|w| is 1.0 in both: we move mass between two elements symmetrically.
    w_prev = torch.tensor([[0.4, 1.6, 1.0, 1.0]])
    w_cur = torch.tensor([[0.6, 1.4, 1.0, 1.0]])
    assert torch.isclose(w_prev.abs().mean(), w_cur.abs().mean())
    s = flips.flip_partition(w_prev, w_cur)
    s.check()
    assert s.weight_only >= 1
    assert s.scale_only == 0


def test_a_flip_either_cause_would_produce_is_redundant():
    """Constructed so the probe weight crosses the boundary *both* ways.

    probe 0.4 -> 0.9 while mean|w| goes 1.0 -> 2/3 (so s: 1.0 -> 1.5):
      actual      state(0.4, 1.0)=0  ->  state(0.9, 1.5)=1     flipped
      weight only state(0.9, 1.0)=1  != 0                      would flip
      scale only  state(0.4, 1.5)=1  != 0                      would flip
    Either cause alone suffices, so it must land in `redundant`, not in a
    single-cause class.
    """
    w_prev = torch.tensor([[0.4, 1.2, 1.2, 1.2]])          # mean|w| = 1.0
    w_cur = torch.tensor([[0.9, 0.5889, 0.5889, 0.5889]])  # mean|w| ~ 2/3
    assert torch.isclose(w_prev.abs().mean(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(w_cur.abs().mean(), torch.tensor(2 / 3), atol=1e-4)
    s = flips.flip_partition(w_prev, w_cur)
    s.check()
    assert s.redundant >= 1


def test_the_partition_is_exact_on_random_tensors():
    """Non-vacuous: each class is computed independently, none is a remainder."""
    torch.manual_seed(0)
    for _ in range(20):
        a = torch.randn(200, 50)
        b = a + torch.randn(200, 50) * 0.35
        s = flips.flip_partition(a, b)
        s.check()                                    # would raise on mismatch
        assert (s.weight_only + s.scale_only + s.redundant + s.joint) == s.flipped


def test_classes_are_never_negative():
    """The failure mode of the residual design this replaced."""
    torch.manual_seed(1)
    for _ in range(20):
        a = torch.randn(120, 40)
        b = a + torch.randn(120, 40) * 0.8
        s = flips.flip_partition(a, b)
        for name in ("weight_only", "scale_only", "redundant", "joint", "cancelled"):
            assert getattr(s, name) >= 0, name


def test_both_reference_conventions_are_computed_and_can_differ():
    """The freeze convention is asymmetric; the card requires reporting both."""
    torch.manual_seed(0)
    a = torch.randn(300, 60)
    b = a + torch.randn(300, 60) * 0.5
    prev = flips.flip_partition(a, b, reference="prev")
    cur = flips.flip_partition(a, b, reference="cur")
    prev.check(); cur.check()
    assert prev.flipped == cur.flipped          # the flip set does not depend on it
    with pytest.raises(ValueError):
        flips.flip_partition(a, b, reference="nonsense")


def test_shape_mismatch_is_an_error_not_a_silent_broadcast():
    with pytest.raises(ValueError, match="shape mismatch"):
        flips.flip_partition(torch.randn(4, 4), torch.randn(4, 8))


def test_stats_add_across_tensors():
    torch.manual_seed(0)
    a, b = torch.randn(50, 20), torch.randn(50, 20)
    x, y = flips.flip_partition(a, b), flips.flip_partition(b, a)
    tot = x + y
    assert tot.n == x.n + y.n and tot.flipped == x.flipped + y.flipped
    tot.check()


def test_check_catches_a_broken_partition():
    """The guard itself must be able to fail, or it is decoration."""
    s = flips.FlipStats(n=10, flipped=5, weight_only=1, scale_only=1,
                        redundant=1, joint=1)
    with pytest.raises(AssertionError, match="partition sums"):
        s.check()


# -- zero occupancy -------------------------------------------------------


def test_zero_occupancy_matches_a_hand_count():
    # mean|w| = 1, so |w·s| < 0.5 rounds to state 0: that is 0.2 and -0.3 only.
    w = torch.tensor([[0.2, -0.3, 1.2, -1.5, 1.4, 1.4]])
    assert torch.isclose(w.abs().mean(), torch.tensor(1.0))
    s = flips.flip_partition(w, w.clone())
    assert s.zero_prev == 2 and s.zero_cur == 2


# -- persistence ----------------------------------------------------------


def _states(seq):
    return [torch.tensor([[v]], dtype=torch.float32) for v in seq]


def test_a_flip_that_reverts_has_zero_persistence():
    # states over time: 1, -1, 1  → the flip at t=1 does not hold at t=2
    held, considered = flips.persistence(_states([1, -1, 1]), k=1)
    assert considered == 1 and held == 0


def test_a_flip_that_sticks_has_full_persistence():
    held, considered = flips.persistence(_states([1, -1, -1, -1]), k=1)
    assert considered >= 1 and held == considered


def test_persistence_counts_are_poolable_not_ratios():
    """Returning (held, considered) rather than a ratio is deliberate: a ratio
    would weight a small tensor the same as a large one when pooled."""
    held, considered = flips.persistence(_states([1, -1, -1, -1]), k=1)
    assert isinstance(held, int) and isinstance(considered, int)


def test_persistence_rejects_k_below_one():
    with pytest.raises(ValueError):
        flips.persistence(_states([1, -1, 1]), k=0)


# -- distance to threshold ------------------------------------------------


def test_distance_histogram_is_self_normalising():
    """mean|w·s| == 1 by construction, which is what makes histograms
    comparable across layers without further normalisation."""
    torch.manual_seed(0)
    for scale in (0.01, 1.0, 100.0):
        w = torch.randn(500, 40) * scale
        scaled = (w * flips.tensor_scale(w)).abs()
        assert torch.isclose(scaled.mean(), torch.tensor(1.0), atol=1e-5)


def test_distance_histogram_puts_mass_where_expected():
    w = torch.tensor([[0.5, 0.5, 1.5, 1.5]])       # all exactly on boundaries
    h = flips.distance_to_threshold(w, bins=4, lo=0.0, hi=2.0)
    assert h.sum() == 4


# -- distance comparators -------------------------------------------------


def test_identical_weights_give_zero_distance_and_unit_cosine():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    d = flips.layer_delta(w, w.clone())
    assert d.l2 == pytest.approx(0.0, abs=1e-5)
    assert d.cosine == pytest.approx(1.0, abs=1e-5)


def test_sign_flipped_weights_give_cosine_minus_one():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    d = flips.layer_delta(w, -w)
    assert d.cosine == pytest.approx(-1.0, abs=1e-5)


def test_cosine_never_exceeds_one_on_large_tensors():
    """fp32 accumulation over millions of elements returned 1.000073 on the real
    360M twins — a value that cannot exist. float64 is why this passes."""
    torch.manual_seed(0)
    a = torch.randn(4096, 960)
    b = a + torch.randn(4096, 960) * 1e-3
    d = flips.layer_delta(a, b)
    assert d.cosine <= 1.0, d.cosine
    assert d.cosine == pytest.approx(1.0, abs=1e-5)
