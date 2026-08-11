"""Phase-2 analysis: do flips predict forgetting better than distance does?

Reads `outputs/phase2/predictors.json` (72 rows) and answers, in order:

1. **The pre-registered gates.** `same` must be ~0 (harness floor); conflict must
   be large in BOTH arms (analytically forced). If either fails, nothing else in
   the run is interpretable and the numbers below are not to be reported.
2. **H1.** Within each arm, across the shift runs, does flip fraction predict
   forgetting better than L2 does — and does any flip-specific term (per-layer
   concentration) add anything L2 cannot?
3. **H2.** Ternary vs float forgetting at matched budget and matched capability.
4. **flips/L2 under task shift**, against phase 1d's diffuse-drift 0.0002. The
   card pre-registered that if this ratio does not move, H1 is refuted as stated.

Correlations are reported with n and with Spearman alongside Pearson: with 3
seeds x 3 budgets x 2 pairs = 18 shift rows per arm, a Pearson r on a
monotone-but-curved relation would overstate the case.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from flab import claims, synthetic

ROOT = Path("outputs/phase2")


def token_acc(arm: str, pair: str, seed: int, cond: str, steps: int,
              task: str) -> float | None:
    """The behavioural witness, read from the probe file it has always been in.

    The retracted H2 claim came from analysing `nll` and never opening `token_acc`
    in the same JSON, across all 72 runs.
    """
    f = (ROOT / f"{arm}-{pair}-s{seed}-B{cond}-{steps}" / "probe-after-0.json")
    if not f.is_file():
        return None
    d = json.loads(f.read_text())["tasks"].get(task)
    return d["token_acc"] if d else None


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    return {"n": int(ok.sum()),
            "pearson": float(stats.pearsonr(x[ok], y[ok]).statistic),
            "spearman": float(stats.spearmanr(x[ok], y[ok]).statistic)}


def r2_of(preds: list[list[float]], y: list[float]) -> float:
    """R^2 of an OLS fit with intercept — used to ask whether a second predictor
    adds anything over the first."""
    X = np.column_stack([np.asarray(p, float) for p in preds] + [np.ones(len(y))])
    y = np.asarray(y, float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def main() -> None:
    rows = json.loads((ROOT / "predictors.json").read_text())
    out: dict = {"n_rows": len(rows)}
    print(f"{len(rows)} rows\n")

    # ---- 1. gates ------------------------------------------------------
    print("=== PRE-REGISTERED GATES ===")
    gates = {}
    for arm in ("ternary", "float"):
        for cond, pair in (("same", None), ("shift", "conflict"), ("shift", "disjoint")):
            sel = [r for r in rows if r["arm"] == arm and r["condition"] == cond
                   and (pair is None or r["pair"] == pair)]
            f = [r["forgetting"] for r in sel if r["forgetting"] is not None]
            label = f"{arm}/{cond}" + (f"/{pair}" if pair else "")
            gates[label] = {"n": len(f), "mean": float(np.mean(f)),
                            "max_abs": float(np.max(np.abs(f)))}
            print(f"  {label:<26} n={len(f):<3} mean {np.mean(f):+8.4f}  "
                  f"max|.| {np.max(np.abs(f)):.4f}")
    out["gates"] = gates
    floor = max(gates[f"{a}/same"]["max_abs"] for a in ("ternary", "float"))
    conflict_ok = all(gates[f"{a}/shift/conflict"]["mean"] > 1.0 for a in ("ternary", "float"))
    print(f"\n  harness floor (max |same|): {floor:.4f} nats")
    print(f"  conflict large in both arms: {conflict_ok}")
    out["harness_floor"] = floor
    out["gate_conflict_large_both_arms"] = bool(conflict_ok)

    # ---- 2. H1 ---------------------------------------------------------
    print("\n=== H1: does flip fraction beat L2? (shift runs only) ===")
    h1 = {}
    for arm in ("ternary", "float"):
        sel = [r for r in rows if r["arm"] == arm and r["condition"] == "shift"
               and r["forgetting"] is not None]
        y = [r["forgetting"] for r in sel]
        flips_ = [r["flip_fraction"] for r in sel]
        l2 = [r["l2"] for r in sel]
        gini = [r["flip_concentration_gini"] for r in sel]
        h1[arm] = {
            "n": len(sel),
            "forgetting_vs_flips": corr(flips_, y),
            "forgetting_vs_l2": corr(l2, y),
            "forgetting_vs_concentration": corr(gini, y),
            "r2_l2_only": r2_of([l2], y),
            "r2_flips_only": r2_of([flips_], y),
            "r2_l2_plus_concentration": r2_of([l2, gini], y),
            "r2_flips_plus_concentration": r2_of([flips_, gini], y),
        }
        d = h1[arm]
        print(f"  {arm}: n={d['n']}")
        for k in ("forgetting_vs_flips", "forgetting_vs_l2", "forgetting_vs_concentration"):
            c = d[k]
            print(f"    {k:<32} pearson {c['pearson']:+.4f}  spearman {c['spearman']:+.4f}")
        print(f"    R2  L2 only {d['r2_l2_only']:.4f} | flips only {d['r2_flips_only']:.4f}"
              f" | L2+conc {d['r2_l2_plus_concentration']:.4f}"
              f" | flips+conc {d['r2_flips_plus_concentration']:.4f}")
    out["h1"] = h1

    # ---- 3. H2 ---------------------------------------------------------
    print("\n=== H2: ternary vs float forgetting, matched budget ===")
    print("    each cell is checked by flab.claims before it may be called a")
    print("    retention difference — see the 2026-08-11 retraction.")
    h2 = {}
    for pair in ("conflict", "disjoint"):
        for steps in (25, 100, 300):
            g = {}
            for arm in ("ternary", "float"):
                f = [r["forgetting"] for r in rows
                     if r["arm"] == arm and r["pair"] == pair
                     and r["condition"] == "shift" and r["b_steps"] == steps
                     and r["forgetting"] is not None]
                g[arm] = (float(np.mean(f)), float(np.std(f, ddof=1)) if len(f) > 1 else None, len(f))
            h2[f"{pair}-{steps}"] = g
            t, fl = g["ternary"], g["float"]

            # The guard: does this cell support a retention claim at all?
            task_a = f"synth-{pair}-a"
            accs = {}
            nll_abs = {}
            for arm in ("ternary", "float"):
                a_vals = [token_acc(arm, pair, sd, "shift", steps, task_a)
                          for sd in (0, 1, 2)]
                a_vals = [v for v in a_vals if v is not None]
                accs[arm] = float(np.mean(a_vals)) if a_vals else None
                base = [r["nll_a_at_A"] for r in json.loads(
                            (ROOT / "results.json").read_text())
                        if r["arm"] == arm and r["pair"] == pair
                        and r["b_steps"] == steps and r["condition"] == "shift"]
                nll_abs[arm] = (float(np.mean(base)) + g[arm][0]) if base else g[arm][0]
            chk = claims.check_forgetting_claim(
                nlls=nll_abs, accuracies=accs,
                chance_nll=synthetic.chance_nll(), accuracy_floor=0.0)
            h2[f"{pair}-{steps}"]["accuracies"] = accs
            h2[f"{pair}-{steps}"]["reportable_as_retention"] = chk.ok
            h2[f"{pair}-{steps}"]["guard_reasons"] = chk.reasons

            flag = "OK" if chk.ok else "NOT A RETENTION CLAIM"
            print(f"  {pair:<9}{steps:>4} steps   ternary {t[0]:+8.4f} (sd {t[1]:.4f})"
                  f"   float {fl[0]:+8.4f} (sd {fl[1]:.4f})"
                  f"   diff {t[0]-fl[0]:+.4f}   acc {accs}  [{flag}]")
            for r in chk.reasons:
                print(f"        ! {r}")
    out["h2"] = h2

    # ---- 4. flips/L2 ---------------------------------------------------
    print("\n=== flips/L2 by condition (phase 1d diffuse drift: 0.000199-0.000208) ===")
    ratios = {}
    for arm in ("ternary", "float"):
        for cond in ("shift", "same"):
            v = [r["flips_per_unit_l2"] for r in rows
                 if r["arm"] == arm and r["condition"] == cond
                 and r["flips_per_unit_l2"] is not None]
            ratios[f"{arm}/{cond}"] = {"n": len(v), "mean": float(np.mean(v)),
                                       "min": float(np.min(v)), "max": float(np.max(v))}
            print(f"  {arm}/{cond:<6} n={len(v):<3} mean {np.mean(v):.6f}  "
                  f"range {np.min(v):.6f}-{np.max(v):.6f}")
    out["flips_per_l2"] = ratios

    print("\n=== scale-driven share (item 22 under task shift) ===")
    sh = [r["scale_share"] for r in rows if r["arm"] == "ternary"
          and r["scale_share"] is not None]
    out["ternary_scale_share"] = {"n": len(sh), "mean": float(np.mean(sh)),
                                  "max": float(np.max(sh))}
    print(f"  ternary: mean {np.mean(sh):.6%}  max {np.max(sh):.6%}  (n={len(sh)})")

    (ROOT / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT / 'analysis.json'}")


if __name__ == "__main__":
    main()
