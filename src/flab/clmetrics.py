"""Continual-learning metrics — ACC, BWT, FWT (phase-1b task 4).

These exist to make our runs comparable with arXiv 2606.27634, whose metrics are
**accuracy-based**. Phase 0 and the 1a shakedown both found accuracy to be a
poor instrument at this scale — most sharply when ScienceQA's NLL worsened by
+0.1101 while its token accuracy moved 0.620 → 0.621, i.e. not at all.

That cannot be resolved by picking a side, so every metric here is computed on
**both** observables and reported side by side (Arley, 2026-08-09). A
systematic divergence is a *finding*, not a nuisance: it would say that
published accuracy-based CL metrics understate forgetting at small scale.

**Sign convention.** All metrics are computed on a "score", where higher is
better. Accuracy is used as-is; NLL is negated. So a negative BWT means
forgetting under *either* observable, and the two are directly comparable
without the reader tracking which way each one points.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class CLMetrics:
    acc: float          # mean final score across tasks
    bwt: float          # backward transfer: negative = forgetting
    fwt: float          # forward transfer vs the pre-training baseline
    n_tasks: int
    observable: str     # "accuracy" or "nll"


def _score(value: float, observable: str) -> float:
    """Higher is better, for both observables."""
    if observable == "accuracy":
        return value
    if observable == "nll":
        return -value          # lower NLL is better
    raise ValueError(f"unknown observable {observable!r}")


def compute(matrix: list[list[float]], baseline: list[float], observable: str) -> CLMetrics:
    """ACC / BWT / FWT from a boundary × task matrix.

    `matrix[k][j]` is task *j*'s value at the boundary after stage *k*, and
    `baseline[j]` is task *j* before any training. Both are raw values in the
    observable's own units; the sign convention is applied here.

    - **ACC** — mean score on every task at the final boundary.
    - **BWT** — `mean_j<K ( a[K][j] − a[j][j] )`: how much each task changed
      between "just after it was trained" and "at the end". Negative is
      forgetting.
    - **FWT** — `mean_j>0 ( a[j-1][j] − baseline[j] )`: how a task stands
      *before* it is trained, relative to the untouched model. Negative means
      training on other tasks actively hurt it.
    """
    n = len(baseline)
    if not matrix or any(len(row) != n for row in matrix):
        raise ValueError("matrix rows must each have one entry per task")
    if len(matrix) != n:
        raise ValueError(f"expected one boundary per task, got {len(matrix)} for {n} tasks")

    a = [[_score(v, observable) for v in row] for row in matrix]
    b = [_score(v, observable) for v in baseline]
    last = n - 1

    acc = sum(a[last]) / n
    # BWT is undefined for a single task: there is no "earlier" task to forget.
    bwt = sum(a[last][j] - a[j][j] for j in range(last)) / last if last else float("nan")
    fwt = sum(a[j - 1][j] - b[j] for j in range(1, n)) / last if last else float("nan")
    return CLMetrics(acc=acc, bwt=bwt, fwt=fwt, n_tasks=n, observable=observable)


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float | None, int]:
    """Pearson r, a two-sided p-value, and n.

    The paper's central quantitative claim is r = −0.497, p < 0.001 between KL
    drift and accuracy. Reproducing the *sign and rough magnitude* is the
    calibration gate; matching r to two decimals on a different model would be
    luck rather than validation.

    The p-value uses the t = r·sqrt((n−2)/(1−r²)) statistic. With the handful of
    boundaries a single run produces, n is small and p should be read as
    indicative — pool across seeds and orders before taking it seriously.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("pearson needs equal-length inputs")
    if n < 3:
        return float("nan"), None, n
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan"), None, n
    r = sxy / math.sqrt(sxx * syy)

    if abs(r) >= 1.0:
        return r, 0.0, n
    t = r * math.sqrt((n - 2) / (1 - r * r))
    try:
        from scipy import stats  # type: ignore

        p = float(2 * stats.t.sf(abs(t), df=n - 2))
    except Exception:
        # Normal approximation; honest for n large, rough for n small — which is
        # why the docstring says to pool before trusting it.
        p = math.erfc(abs(t) / math.sqrt(2))
    return r, p, n
