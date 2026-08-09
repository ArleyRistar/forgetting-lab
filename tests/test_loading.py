"""Tests for the shared converted-model load path (open item 21).

The property under test is the one that would fail silently: loading a ternary
checkpoint without re-applying BitLinear yields a float model that scores
perfectly well and is wrong. So the tests assert on *effective* weights and on
the guard rejecting a float model, not on the happy path alone.

Numerics are built by hand rather than through a real checkpoint — the phase-1b
KL bug survived a model-based test because a small model's outputs hide almost
everything.
"""
import json

import pytest
import torch
import torch.nn as nn

from flab import bitlinear as bl
from flab import loading


# -- checkpoint detection -------------------------------------------------


def _write_meta(d, mode="ternary", warmup_completed=True, final_lambda=1.0):
    (d / "convert.json").write_text(json.dumps(
        {"config": {"mode": mode}, "warmup_completed": warmup_completed,
         "final_lambda": final_lambda}))


def test_detects_a_ternary_checkpoint(tmp_path):
    _write_meta(tmp_path)
    assert loading.is_ternary_checkpoint(tmp_path)


def test_detects_metadata_in_the_parent_directory(tmp_path):
    """`save_model` writes weights to `final/`; the metadata sits one level up."""
    _write_meta(tmp_path)
    final = tmp_path / "final"
    final.mkdir()
    assert loading.is_ternary_checkpoint(final)


def test_float_checkpoint_is_not_ternary(tmp_path):
    _write_meta(tmp_path, mode="float")
    assert not loading.is_ternary_checkpoint(tmp_path)


def test_missing_metadata_is_not_ternary(tmp_path):
    assert not loading.is_ternary_checkpoint(tmp_path)


def test_directory_name_does_not_make_a_checkpoint_ternary(tmp_path):
    """A path called ternary-360m proves nothing; the run's own record does."""
    d = tmp_path / "ternary-360m"
    d.mkdir()
    assert not loading.is_ternary_checkpoint(d)


def test_incomplete_warmup_raises_rather_than_loading_quietly(tmp_path):
    """A ternary run that never reached lambda=1 is not a ternary model, and
    reporting it as one is exactly the mislabelling this module exists to stop."""
    _write_meta(tmp_path, warmup_completed=False, final_lambda=0.6)
    with pytest.raises(ValueError, match="warmup never completed"):
        loading.is_ternary_checkpoint(tmp_path)


# -- the guard ------------------------------------------------------------


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.down_proj = nn.Linear(16, 16)


def test_assert_ternary_accepts_a_converted_model():
    torch.manual_seed(0)
    m, n = bl.convert(_Tiny(), lambda_=1.0)
    assert n == 2
    assert loading.assert_ternary(m) == 2


def test_assert_ternary_rejects_a_half_warmed_model():
    """lambda=0.5 means the forward pass is not ternary, whatever the weights."""
    m, _ = bl.convert(_Tiny(), lambda_=0.5)
    with pytest.raises(AssertionError, match="not fully ternary"):
        loading.assert_ternary(m)


def test_assert_ternary_reports_zero_for_an_unconverted_model():
    """The dangerous case: a float model passes vacuously, so callers must check
    the count rather than the absence of an exception."""
    assert loading.assert_ternary(_Tiny()) == 0


def test_assert_ternary_catches_continuous_effective_weights(monkeypatch):
    """If quantisation silently stopped ternarising, the guard must fire."""
    m, _ = bl.convert(_Tiny(), lambda_=1.0)
    monkeypatch.setattr(bl, "weight_quant", lambda w: w)   # sabotage
    with pytest.raises(AssertionError, match="distinct values"):
        loading.assert_ternary(m)


def test_effective_weights_really_are_three_valued_after_convert():
    """Ties the guard to the actual quantiser rather than to itself."""
    torch.manual_seed(0)
    m, _ = bl.convert(_Tiny(), lambda_=1.0)
    for mod in m.modules():
        if isinstance(mod, bl.BitLinear):
            assert torch.unique(bl.weight_quant(mod.weight)).numel() <= 3
