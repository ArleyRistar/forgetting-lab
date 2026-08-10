"""Weight-state flips for ternary layers (spec §6 1d, design card 2026-08-10).

The quantity that flips is the **state** `σ ∈ {-1, 0, +1}`, not the effective
value. Effective values are `{-m, 0, +m}` with `m = mean|W|`, and `m` moves every
step, so every weight's effective *value* changes constantly and a metric built
on value change measures nothing. (Spec §6 1d says "effective-value change"; read
literally that is degenerate — this is item 18's rename made real.)

**Why the partition exists.** The absmean scale is per-tensor and recomputed every
forward pass (`bitlinear.py:40`), so any fine-tuning moves the decision boundary
and reclassifies every near-threshold weight at once. A raw flip count therefore
partly measures *amount of training*, which correlates with forgetting for
trivial reasons — H1 could confirm itself on a model that learned nothing. So
every flip is assigned to exactly one of four disjoint classes by asking what
each cause would have done *alone*:

    weight-only   weight motion alone would flip it; scale motion alone would not
    scale-only    scale motion alone would flip it; weight motion alone would not
    redundant     either alone would have flipped it
    joint         neither alone flips it — it needs both

These four are disjoint and sum **exactly** to the flip count. An earlier design
used `interaction = total - weight - scale`, which can go negative (both causes
flip a weight but their combined motion cancels) and whose "total = sum of parts"
test could not fail, being true by construction.

Everything is per-tensor because the scale is per-tensor. Callers aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import torch

# Which endpoint the counterfactuals are measured against.
REFERENCES = ("prev", "cur")


def tensor_scale(w: torch.Tensor) -> torch.Tensor:
    """`1 / mean|w|`, matching `bitlinear.weight_quant` including its clamp."""
    return 1.0 / w.abs().mean().clamp(min=1e-5)


def state(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Ternary state in {-1, 0, +1}.

    Uses the same ops as `weight_quant` rather than re-deriving with `>= 0.5`
    comparisons: `torch.round` is round-half-to-even, so a reimplementation
    disagrees with the forward pass at exact ties. Tested against
    `sign(weight_quant(w))`, ties included.
    """
    return (w * scale).round().clamp(-1, 1)


@dataclass
class FlipStats:
    """Counts for one tensor over one interval. All counts, not fractions —
    fractions of different-sized tensors cannot be averaged."""

    n: int = 0
    flipped: int = 0
    weight_only: int = 0
    scale_only: int = 0
    redundant: int = 0
    joint: int = 0
    cancelled: int = 0          # a counterfactual would flip, the real motion did not
    zero_prev: int = 0
    zero_cur: int = 0

    def __add__(self, other: "FlipStats") -> "FlipStats":
        return FlipStats(**{k: getattr(self, k) + getattr(other, k)
                            for k in self.__dataclass_fields__})

    @property
    def flip_fraction(self) -> float:
        return self.flipped / self.n if self.n else 0.0

    def check(self) -> None:
        """The partition must be exact. Cheap enough to assert every time."""
        parts = self.weight_only + self.scale_only + self.redundant + self.joint
        if parts != self.flipped:
            raise AssertionError(
                f"partition sums to {parts} but {self.flipped} flipped — the "
                "four classes are meant to be disjoint and exhaustive")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["flip_fraction"] = self.flip_fraction
        return d


def flip_partition(w_prev: torch.Tensor, w_cur: torch.Tensor, *,
                   reference: str = "prev") -> FlipStats:
    """Partition the flips between two checkpoints of one tensor.

    `reference` picks which endpoint the counterfactuals are compared against.
    The convention is genuinely asymmetric, so the card requires reporting both;
    neither is more correct, and the pair bounds how much the choice matters.
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}")
    if w_prev.shape != w_cur.shape:
        raise ValueError(f"shape mismatch: {w_prev.shape} vs {w_cur.shape}")

    w_prev = w_prev.float()
    w_cur = w_cur.float()
    s_prev, s_cur = tensor_scale(w_prev), tensor_scale(w_cur)

    st_prev = state(w_prev, s_prev)
    st_cur = state(w_cur, s_cur)
    # The two counterfactuals: move one cause at a time.
    st_weight_moved = state(w_cur, s_prev)     # weights moved, scale held
    st_scale_moved = state(w_prev, s_cur)      # scale moved, weights held

    base = st_prev if reference == "prev" else st_cur
    flipped = st_cur != st_prev
    weight_alone = st_weight_moved != base
    scale_alone = st_scale_moved != base

    both = weight_alone & scale_alone
    return FlipStats(
        n=w_prev.numel(),
        flipped=int(flipped.sum()),
        weight_only=int((flipped & weight_alone & ~scale_alone).sum()),
        scale_only=int((flipped & scale_alone & ~weight_alone).sum()),
        redundant=int((flipped & both).sum()),
        joint=int((flipped & ~weight_alone & ~scale_alone).sum()),
        cancelled=int(((weight_alone | scale_alone) & ~flipped).sum()),
        zero_prev=int((st_prev == 0).sum()),
        zero_cur=int((st_cur == 0).sum()),
    )


def persistence(states: list[torch.Tensor], k: int) -> tuple[int, int]:
    """Of the weights that flipped at each step, how many hold the new state
    `k` intervals later.

    Returns `(held, flips_considered)` so callers can pool across tensors — a
    ratio of ratios would weight a 921k-element tensor the same as a 3.7M one.

    Answers spec §12 open question 4 for the window-length half: `k` is in
    checkpoint intervals, and the caller decides the cadence those represent.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    held = considered = 0
    for t in range(1, len(states) - k):
        flipped = states[t] != states[t - 1]
        still = (states[t + k] == states[t]) & flipped
        considered += int(flipped.sum())
        held += int(still.sum())
    return held, considered


def distance_to_threshold(w: torch.Tensor, bins: int = 40,
                          lo: float = 0.0, hi: float = 2.0) -> torch.Tensor:
    """Histogram of `|w · s|`, whose decision boundary sits at 0.5.

    Scaled units are self-normalising — `mean|w · s| = 1` by construction — so
    histograms are directly comparable across layers and checkpoints without
    any further normalisation. Mass piling at 0.5 is Tequila's "deadzone
    boundary"; mass at 1.5 is the ±1 clamp edge and is inert.
    """
    scaled = (w.float() * tensor_scale(w.float())).abs()
    return torch.histc(scaled, bins=bins, min=lo, max=hi)


@dataclass
class LayerDelta:
    """Distance comparators — the predictors H1 claims flips beat.

    Computed on **both** arms: measuring distance only on the float twin would
    confound the predictor with the arm, so a flips-beat-distance claim would
    not be testable within the ternary arm at all (item 19).
    """

    l2: float = 0.0
    cosine: float = 0.0
    rel_l2: float = 0.0
    n: int = 0


def layer_delta(w_prev: torch.Tensor, w_cur: torch.Tensor) -> LayerDelta:
    # float64: in fp32 the dot product over millions of elements accumulates
    # enough error to return cosines slightly ABOVE 1 (measured 1.000073 on the
    # 360M twins), which is not a rounding curiosity but a number that cannot
    # exist and would discredit any plot it appeared in.
    a, b = w_prev.double().flatten(), w_cur.double().flatten()
    d = b - a
    denom = a.norm() * b.norm()
    return LayerDelta(
        l2=float(d.norm()),
        cosine=float((a @ b) / denom) if denom > 0 else 0.0,
        rel_l2=float(d.norm() / a.norm()) if a.norm() > 0 else 0.0,
        n=a.numel(),
    )
