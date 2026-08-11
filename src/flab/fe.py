"""Two-way fixed-effects estimate of a residual trace (phase 2b).

Fits, over the non-v2 letters of each key:

    log p[key, letter] = key_effect + letter_effect + gamma_v1 * 1[letter == v1]
                                                   + gamma_vp * 1[letter == v']

The key effect absorbs each key's difficulty and its total non-v2 mass — which
is exactly the B-mastery level that produced the retracted "2 of 6 nats" artefact
— so the estimate is internally referenced per key and cannot be moved by one arm
learning B harder. The letter effect absorbs the fact that the eight values are
not exchangeable: every letter is a trained answer for many other keys, so a raw
across-letter mean measures letter marginals, not memory.

Inference is at the **seed** level. The seven log-probs within a key are
mechanically dependent and every cell in a run shares one model, so cell-level
OLS standard errors are not credible. `paired_contrast` returns one number per
key, and the caller aggregates across seeds on t with 2 df.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FEResult:
    gamma_v1: float
    gamma_vp: float
    n_obs: int
    n_keys: int
    n_letters: int


def fit(logp: dict[tuple[str, str], float], v1: dict[str, str],
        vp: dict[str, str], v2: dict[str, str]) -> FEResult:
    """`logp[(key, letter)]` over letters excluding each key's v2."""
    keys = sorted({k for k, _ in logp})
    letters = sorted({l for _, l in logp})
    ki = {k: i for i, k in enumerate(keys)}
    li = {l: i for i, l in enumerate(letters)}

    rows, y = [], []
    for (k, l), val in logp.items():
        if l == v2.get(k):
            continue                       # v2 is excluded by construction
        r = np.zeros(len(keys) + len(letters) + 2)
        r[ki[k]] = 1.0
        r[len(keys) + li[l]] = 1.0
        r[-2] = 1.0 if l == v1.get(k) else 0.0
        r[-1] = 1.0 if l == vp.get(k) else 0.0
        rows.append(r)
        y.append(val)

    X = np.asarray(rows)
    beta, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
    return FEResult(gamma_v1=float(beta[-2]), gamma_vp=float(beta[-1]),
                    n_obs=len(y), n_keys=len(keys), n_letters=len(letters))


def paired_contrast(logp_a: dict[tuple[str, str], float],
                    logp_p: dict[tuple[str, str], float],
                    v1: dict[str, str], v2: dict[str, str]) -> dict[str, float]:
    """Per-key contrast: v1's letter-demeaned residual in A→B minus in placebo→B.

    Returned per key rather than pooled so the caller can take a key-level SE.
    Key idiosyncrasies shared across the two conditions cancel by construction,
    which a pooled fit does not guarantee.

    **Bias-corrected by n/(n-1).** The v1 cell is part of the key mean it is
    measured against, so a true gamma comes back as gamma*(1 - 1/n) — with 7
    non-v2 letters that is a 14% underestimate, caught by the planted-effect test
    and large enough to matter against a 3-SE threshold.
    """
    def residual(logp: dict[tuple[str, str], float]) -> dict[str, float]:
        by_letter: dict[str, list[float]] = {}
        for (k, l), val in logp.items():
            if l != v2.get(k):
                by_letter.setdefault(l, []).append(val)
        letter_mean = {l: float(np.mean(v)) for l, v in by_letter.items()}
        out = {}
        for k in {k for k, _ in logp}:
            cells = [(l, val) for (kk, l), val in logp.items()
                     if kk == k and l != v2.get(k)]
            if not cells or k not in v1:
                continue
            adj = {l: val - letter_mean[l] for l, val in cells}
            key_mean = float(np.mean(list(adj.values())))
            if v1[k] in adj:
                n = len(adj)
                shrink = (n - 1) / n if n > 1 else 1.0
                out[k] = (adj[v1[k]] - key_mean) / shrink
        return out

    ra, rp = residual(logp_a), residual(logp_p)
    return {k: ra[k] - rp[k] for k in set(ra) & set(rp)}


def seed_level(contrasts: list[float]) -> dict[str, float]:
    """Mean +- sd/sqrt(n) over per-seed contrasts, on t with n-1 df.

    Three seeds means t(2), where 95% needs 4.303 x SE rather than 1.96 — the
    card fixes this in advance so it cannot be renegotiated after the numbers.
    """
    a = np.asarray(contrasts, float)
    n = len(a)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    crit = {2: 4.303, 3: 3.182, 4: 2.776}.get(n - 1, 1.96)
    return {"n_seeds": n, "mean": mean, "se": se,
            "t": mean / se if se else float("nan"),
            "ci95_lo": mean - crit * se, "ci95_hi": mean + crit * se,
            "crit_t": crit}
