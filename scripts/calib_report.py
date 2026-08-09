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
        "model": cfg["model"].split("/")[-1],
        # Runs from different measurement regimes must never be pooled: the
        # flab-style runs used our tags, answer-token KL and full-sequence loss.
        # Averaging them with replication runs would produce numbers that are
        # not measurements of anything.
        "style": cfg.get("prompt_style", "flab"),
        "kl_scope": cfg.get("kl_scope", "answer_tokens"),
    }


def acc_of(task_probe: dict) -> float:
    """Accuracy to use for CL metrics.

    Prefer `content_acc` — accuracy excluding each example's final answer token.
    Raw `token_acc` includes the assistant turn terminator, which swings 50
    points on FOMC purely from answer-length effects between tasks and would
    make every pooled number a measure of formatting rather than knowledge.
    Falls back to token_acc only when there is no content to score.
    """
    ca = task_probe.get("content_acc")
    return ca if ca is not None else task_probe["token_acc"]


def op_curve(run) -> list[float]:
    """Their OP: mean accuracy over tasks *seen so far*, per checkpoint."""
    out = []
    for k, b in enumerate(run["bounds"]):
        seen = run["order"][: k + 1]
        out.append(st.mean(acc_of(b["tasks"][t]) for t in seen if t in b["tasks"]))
    return out


def metrics(run, observable, key):
    if key == "content_acc":
        mat = [[acc_of(b["tasks"][t]) for t in run["tasks"]] for b in run["bounds"]]
        base = [acc_of(run["base"]["tasks"][t]) for t in run["tasks"]]
    else:
        mat = [[b["tasks"][t][key] for t in run["tasks"]] for b in run["bounds"]]
        base = [run["base"]["tasks"][t][key] for t in run["tasks"]]
    return clmetrics.compute(mat, base, observable)


def main() -> None:
    globs = list(Path("outputs/runs").glob("calib-*")) + list(Path("outputs/runs").glob("repl-*"))
    every = sorted(filter(None, (load_run(d) for d in globs)),
                   key=lambda r: (r["order"][0], r["seed"]))
    if not every:
        print("no complete calibration runs yet")
        return

    want = sys.argv[1] if len(sys.argv) > 1 else "paper"
    runs = [r for r in every if r["style"] == want]
    skipped = len(every) - len(runs)
    if not runs:
        print(f"no complete runs with prompt_style={want!r} "
              f"({skipped} run(s) in other regimes, not pooled)")
        return

    print(f"{len(runs)} complete run(s) with prompt_style={want!r}")
    if skipped:
        print(f"  ({skipped} run(s) from other measurement regimes excluded — "
              "pooling across styles would not measure anything)")
    print()
    print(f"{'run':<26}{'seed':>5}{'order':>8}"
          f"{'ACC(acc)':>10}{'BWT(acc)':>10}{'ACC(nll)':>10}{'BWT(nll)':>10}{'KL_final':>10}")
    print("-" * 89)

    rows, kl_acc_pairs = [], []
    for r in runs:
        ma = metrics(r, "accuracy", "content_acc")
        mn = metrics(r, "nll", "nll")
        kl_last = r["bounds"][-1].get("stability", {}).get("kl_from_base")  # KL(cur||base), the paper's
        order = "fwd" if r["order"][0] == "FOMC" else "rev"
        rows.append((order, ma, mn, kl_last))
        print(f"{r['name']:<26}{r['seed']:>5}{order:>8}"
              f"{ma.acc:>10.4f}{ma.bwt:>10.4f}{mn.acc:>10.4f}{mn.bwt:>10.4f}"
              f"{(f'{kl_last:.4f}' if kl_last is not None else 'n/a'):>10}")

        # Every boundary contributes a (KL, mean-accuracy) point.
        for b in r["bounds"]:
            kl = b.get("stability", {}).get("kl_from_base")
            if kl is not None:
                acc = st.mean(acc_of(b["tasks"][t]) for t in r["tasks"])
                kl_acc_pairs.append((kl, acc))

    def pooled(label, vals):
        if len(vals) < 2:
            return f"{label:<22}{vals[0]:>10.4f}{'  (n=1, no sd)':>16}" if vals else ""
        return (f"{label:<22}{st.mean(vals):>10.4f}  ± {st.stdev(vals):<8.4f}"
                f"n={len(vals)}")

    # Their OP curve, directly comparable to the paper's table.
    PAPER_OP = {"Llama-3.2-1B-Instruct": [0.530, 0.647, 0.485],
                "Qwen3.5-0.8B": [0.533, 0.670, 0.591],
                "gemma-3-1b-it": [0.547, 0.487, 0.320]}
    print("\nOP curve (mean content_acc over tasks seen so far) vs the paper")
    for m in sorted({r_["model"] for r_ in runs}):
        sel = [op_curve(r_) for r_ in runs if r_["model"] == m and r_["order"][0] == "FOMC"]
        if not sel:
            continue
        ours = [st.mean(x[i] for x in sel) for i in range(len(sel[0]))]
        theirs = PAPER_OP.get(m)
        o = "  ".join(f"{v:.3f}" for v in ours)
        t_ = "  ".join(f"{v:.3f}" for v in theirs) if theirs else "n/a"
        print(f"  {m:<26} ours {o}   paper {t_}   (n={len(sel)} seed(s))")

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
        print(f"\n1. WITHIN-run KL vs accuracy: r={r:+.3f}  p={ps}  n={n}")
        print("   (expected positive: both rise with training. NOT the paper's"
              " claim — see the cross-model figure above.)")
        print(f"   paper: r={PAPER_R:+.3f}, p<0.001")
        same_sign = (r < 0) == (PAPER_R < 0)
        print(f"   -> sign {'MATCHES' if same_sign else 'DIFFERS'}; "
              f"|r| {'comparable' if 0.5 * abs(PAPER_R) <= abs(r) <= 2 * abs(PAPER_R) else 'differs in magnitude'}")
    else:
        print("\n1. KL vs accuracy: too few paired boundaries")

    # -- the cross-model gate, which is what the paper's r actually measures --
    # Their r = -0.497 is "across all models": stable models drift less and
    # score higher. Within a single model, KL and accuracy both rise with
    # training, so a within-run correlation measures the shared time trend and
    # comes out positive regardless. Only the cross-model pooling tests them.
    by_model = {}
    for r_ in runs:
        for i, b in enumerate(r_["bounds"]):
            kl = b.get("stability", {}).get("kl_from_base")
            if kl is None:
                continue
            acc = st.mean(acc_of(b["tasks"][t]) for t in r_["tasks"])
            by_model.setdefault(r_["model"], []).append((i, kl, acc))

    if len(by_model) >= 2:
        print("\nPer model (final checkpoint, pooled over seeds and orders):")
        print(f"  {'model':<26}{'KL':>18}{'accuracy':>18}")
        summary = []
        for m, pts in sorted(by_model.items()):
            last = max(i for i, _, _ in pts)
            kls = [k for i, k, _ in pts if i == last]
            accs = [a for i, _, a in pts if i == last]
            summary.append((m, st.mean(kls), st.mean(accs)))
            sd_k = f"± {st.stdev(kls):.4f}" if len(kls) > 1 else ""
            sd_a = f"± {st.stdev(accs):.4f}" if len(accs) > 1 else ""
            print(f"  {m:<26}{st.mean(kls):>10.4f} {sd_k:<7}{st.mean(accs):>10.4f} {sd_a:<7}")

        pooled_kl = [k for pts in by_model.values() for _, k, _ in pts]
        pooled_acc = [a for pts in by_model.values() for _, _, a in pts]
        rr, pp, nn = clmetrics.pearson(pooled_kl, pooled_acc)
        print(f"\n  CROSS-MODEL KL vs accuracy: r={rr:+.3f} p="
              f"{'n/a' if pp is None else f'{pp:.5f}'} n={nn}")
        print(f"  paper: r={PAPER_R:+.3f}, p<0.001  ->  sign "
              f"{'MATCHES' if (rr < 0) == (PAPER_R < 0) else 'DIFFERS'}")

        print("\n  Predicted ordering (theirs): gemma drifts most/scores worst,"
              " qwen least/best")
        by_kl = [m for m, _, _ in sorted(summary, key=lambda x: -x[1])]
        by_acc = [m for m, _, _ in sorted(summary, key=lambda x: x[2])]
        print(f"    ours by KL desc : {' > '.join(by_kl)}")
        print(f"    ours by acc asc : {' < '.join(by_acc)}")
        print(f"    -> orderings {'AGREE' if by_kl == by_acc else 'DISAGREE'}"
              " with each other (they must, if drift tracks damage)")

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
