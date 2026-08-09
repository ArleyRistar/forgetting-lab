#!/usr/bin/env python3
"""Pool the phase-1b calibration runs and state the verdict (task 6).

The gate is NOT "do our absolute numbers match theirs" — different model, and
matching an accuracy across models would be luck. It is:

  1. Does **KL drift correlate negatively with accuracy**? Their central claim
     is r = -0.497, p < 0.001. Sign and rough magnitude is the bar.
  2. Does the **KL ~ 0.8 instability threshold** behave as claimed, and is it
     really order-independent? The reversed-order arm exists to test that rather
     than inherit it.
  3. Do ACC / BWT / FWT come out with sensible signs on both observables?

Every metric is computed on accuracy *and* NLL, and disagreement between them is
reported as a finding rather than averaged away (Arley, 2026-08-09).

Stdlib only, so it runs under system python3 while .venv is busy.

Usage: python3 scripts/calib_report.py
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flab import clmetrics  # noqa: E402

PAPER_R = -0.497
KL_THRESHOLD = 0.8


def load_run(d: Path):
    state = json.loads((d / "runstate.json").read_text())
    if not state.get("baseline_probe"):
        return None
    if any(s.get("probe") is None for s in state["stages"]):
        return None                      # incomplete; skip rather than half-report
    base = json.loads((d / state["baseline_probe"]).read_text())
    bounds = [json.loads((d / s["probe"]).read_text()) for s in state["stages"]]
    tasks = list(base["tasks"])
    cfg = json.loads((d / "run.json").read_text())["config"]
    return {
        "name": d.name, "tasks": tasks, "base": base, "bounds": bounds,
        "seed": cfg["seed"], "order": [s["task"] for s in cfg["stages"]],
    }


def metrics(run, observable, key):
    mat = [[b["tasks"][t][key] for t in run["tasks"]] for b in run["bounds"]]
    base = [run["base"]["tasks"][t][key] for t in run["tasks"]]
    return clmetrics.compute(mat, base, observable)


def main() -> None:
    runs = sorted(filter(None, (load_run(d) for d in Path("outputs/runs").glob("calib-*"))),
                  key=lambda r: (r["order"][0], r["seed"]))
    if not runs:
        print("no complete calibration runs yet")
        return

    print(f"{len(runs)} complete run(s)\n")
    print(f"{'run':<26}{'seed':>5}{'order':>8}"
          f"{'ACC(acc)':>10}{'BWT(acc)':>10}{'ACC(nll)':>10}{'BWT(nll)':>10}{'KL_final':>10}")
    print("-" * 89)

    rows, kl_acc_pairs = [], []
    for r in runs:
        ma = metrics(r, "accuracy", "token_acc")
        mn = metrics(r, "nll", "nll")
        kl_last = r["bounds"][-1].get("stability", {}).get("kl_from_base")
        order = "fwd" if r["order"][0] == "FOMC" else "rev"
        rows.append((order, ma, mn, kl_last))
        print(f"{r['name']:<26}{r['seed']:>5}{order:>8}"
              f"{ma.acc:>10.4f}{ma.bwt:>10.4f}{mn.acc:>10.4f}{mn.bwt:>10.4f}"
              f"{(f'{kl_last:.4f}' if kl_last is not None else 'n/a'):>10}")

        # Every boundary contributes a (KL, mean-accuracy) point.
        for b in r["bounds"]:
            kl = b.get("stability", {}).get("kl_from_base")
            if kl is not None:
                acc = st.mean(b["tasks"][t]["token_acc"] for t in r["tasks"])
                kl_acc_pairs.append((kl, acc))

    def pooled(label, vals):
        if len(vals) < 2:
            return f"{label:<22}{vals[0]:>10.4f}{'  (n=1, no sd)':>16}" if vals else ""
        return (f"{label:<22}{st.mean(vals):>10.4f}  ± {st.stdev(vals):<8.4f}"
                f"n={len(vals)}")

    print("\nPooled across seeds")
    for order in ("fwd", "rev"):
        sel = [(ma, mn, kl) for o, ma, mn, kl in rows if o == order]
        if not sel:
            continue
        print(f"\n  {order} order:")
        print("   ", pooled("ACC (accuracy)", [m.acc for m, _, _ in sel]))
        print("   ", pooled("BWT (accuracy)", [m.bwt for m, _, _ in sel]))
        print("   ", pooled("ACC (nll)", [m.acc for _, m, _ in sel]))
        print("   ", pooled("BWT (nll)", [m.bwt for _, m, _ in sel]))
        kls = [k for _, _, k in sel if k is not None]
        if kls:
            print("   ", pooled("final KL", kls))

    # -- the gate ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("CALIBRATION GATE")

    if len(kl_acc_pairs) >= 3:
        r, p, n = clmetrics.pearson([x for x, _ in kl_acc_pairs],
                                    [y for _, y in kl_acc_pairs])
        ps = "n/a" if p is None else f"{p:.5f}"
        print(f"\n1. KL vs accuracy: r={r:+.3f}  p={ps}  n={n}")
        print(f"   paper: r={PAPER_R:+.3f}, p<0.001")
        same_sign = (r < 0) == (PAPER_R < 0)
        print(f"   -> sign {'MATCHES' if same_sign else 'DIFFERS'}; "
              f"|r| {'comparable' if 0.5 * abs(PAPER_R) <= abs(r) <= 2 * abs(PAPER_R) else 'differs in magnitude'}")
    else:
        print("\n1. KL vs accuracy: too few paired boundaries")

    kls_fwd = [k for o, _, _, k in rows if o == "fwd" and k is not None]
    kls_rev = [k for o, _, _, k in rows if o == "rev" and k is not None]
    if kls_fwd and kls_rev:
        print(f"\n2. KL ~ {KL_THRESHOLD} threshold, order-independence:")
        print(f"   fwd final KL {st.mean(kls_fwd):.4f}   rev final KL {st.mean(kls_rev):.4f}")
        above = [k for k in kls_fwd + kls_rev if k > KL_THRESHOLD]
        print(f"   {len(above)}/{len(kls_fwd) + len(kls_rev)} runs exceed {KL_THRESHOLD}")
    else:
        print(f"\n2. KL ~ {KL_THRESHOLD} threshold: need both orders")

    print("\n3. Observable agreement:")
    dis = [(o, ma.bwt, mn.bwt) for o, ma, mn, _ in rows if (ma.bwt < 0) != (mn.bwt < 0)]
    if dis:
        print(f"   *** {len(dis)}/{len(rows)} runs DISAGREE on the sign of BWT ***")
        for o, a, n_ in dis:
            print(f"     {o}: accuracy {a:+.4f} vs nll {n_:+.4f}")
        print("   That is a result about the metrics, not noise to average.")
    else:
        print(f"   all {len(rows)} runs agree on the sign of BWT across observables")


if __name__ == "__main__":
    main()
