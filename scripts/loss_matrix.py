#!/usr/bin/env python3
"""Assemble a run's boundary probes into the loss matrix (phase-1a task 6).

Rows are boundaries (baseline, then after each stage); columns are tasks. The
cell is held-out answer-token NLL. Everything the phase-1 question asks for
falls out of this one table:

  * **Forgetting** — a task's NLL rising after training moved on to later tasks.
  * **Backward transfer** — an earlier task improving because of a later one.
  * **Forward transfer** — a not-yet-trained task improving anyway.

Deltas are always against **that task's own baseline**, never across tasks. The
dev trio spans 1 answer token (FOMC) to ~200 (ScienceQA), so a cross-task
comparison would track answer length rather than anything about the model —
and spec §9 needs the same discipline later for the ternary/float pair, whose
absolute losses are not comparable by construction.

Stdlib only, so it runs under system python3 without touching .venv while a
run is using it.

Usage: python3 scripts/loss_matrix.py [outputs/runs/dev-3stage]
"""
import json
import sys
from pathlib import Path

# clmetrics is stdlib-only by design (scipy is an optional import inside it), so
# this stays runnable under system python3 while a run is using .venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flab import clmetrics  # noqa: E402


def load(run_dir: Path) -> tuple[list[str], list[tuple[str, dict]]]:
    state = json.loads((run_dir / "runstate.json").read_text())
    rows = []
    if state.get("baseline_probe"):
        rows.append(("baseline", json.loads((run_dir / state["baseline_probe"]).read_text())))
    for i, s in enumerate(state["stages"]):
        if s.get("probe") and (run_dir / s["probe"]).is_file():
            rows.append((f"after {s['name']}", json.loads((run_dir / s["probe"]).read_text())))
    tasks = list(rows[0][1]["tasks"]) if rows else []
    return tasks, rows


def table(title: str, tasks, rows, cell) -> None:
    print(f"\n{title}")
    print(f"{'boundary':<18}" + "".join(f"{t:>13}" for t in tasks))
    print("-" * (18 + 13 * len(tasks)))
    for label, data in rows:
        print(f"{label:<18}" + "".join(f"{cell(data, t, label):>13}" for t in tasks))


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/dev-3stage")
    tasks, rows = load(run_dir)
    if not rows:
        print(f"no probe files in {run_dir} yet")
        return

    base = {t: rows[0][1]["tasks"][t]["nll"] for t in tasks}

    table("Held-out answer-token NLL", tasks, rows,
          lambda d, t, _: f"{d['tasks'][t]['nll']:.4f}")
    table("Delta vs that task's own baseline (+ = worse = forgetting)", tasks, rows,
          lambda d, t, lab: "  —" if lab == "baseline"
          else f"{d['tasks'][t]['nll'] - base[t]:+.4f}")
    table("Token accuracy", tasks, rows,
          lambda d, t, _: f"{d['tasks'][t]['token_acc']:.3f}")

    first = rows[0][1]["tasks"]
    print("\nScored tokens per task (why NLL is never averaged across them):")
    for t in tasks:
        print(f"  {t:<12}{first[t]['n_tokens']:>7} tokens over {first[t]['n_examples']} examples"
              f"   ({first[t]['n_prompt_truncated']} prompts truncated)")

    warned = [(lab, d["warnings"]) for lab, d in rows if d.get("warnings")]
    if warned:
        print("\nWARNINGS — these boundaries did not measure what they claim:")
        for lab, w in warned:
            print(f"  {lab}: {w}")
    else:
        print("\nNo probe warnings: every boundary measured what it claims.")

    # -- continual-learning metrics, on BOTH observables ------------------
    stages = rows[1:]
    if len(stages) == len(tasks) >= 1:
        print("\nContinual-learning metrics (higher score = better; "
              "negative BWT = forgetting)")
        print(f"{'observable':<14}{'ACC':>10}{'BWT':>10}{'FWT':>10}")
        print("-" * 44)
        out = {}
        for obs, key in (("accuracy", "token_acc"), ("nll", "nll")):
            mat = [[d["tasks"][t][key] for t in tasks] for _, d in stages]
            base = [rows[0][1]["tasks"][t][key] for t in tasks]
            m = clmetrics.compute(mat, base, obs)
            out[obs] = m
            print(f"{obs:<14}{m.acc:>10.4f}{m.bwt:>10.4f}{m.fwt:>10.4f}")

        # The paper's metrics are accuracy-based; phase 0 and the 1a shakedown
        # both found accuracy a poor instrument at this scale. A systematic
        # disagreement here is a finding, not a nuisance.
        a, n = out["accuracy"], out["nll"]
        if (a.bwt < 0) != (n.bwt < 0):
            print("\n  *** OBSERVABLES DISAGREE ON THE SIGN OF BWT ***")
            print("  accuracy and NLL point opposite ways about whether forgetting")
            print("  happened at all. That is a result to write up, not to average.")
        elif abs(a.bwt) < 0.1 * abs(n.bwt):
            print("\n  NOTE: accuracy-BWT is <10% the magnitude of NLL-BWT — the")
            print("  accuracy metric is understating forgetting, as at ScienceQA")
            print("  in the 1a shakedown (NLL +0.110, accuracy 0.620 -> 0.621).")

    # -- drift, and the paper's central correlation -----------------------
    kls = [(lab, d["stability"]["kl_from_base"]) for lab, d in rows
           if d.get("stability", {}).get("kl_from_base") is not None]
    if kls:
        print("\nReference-set drift (KL from base, nats/token)")
        for lab, kl in kls:
            flag = "  <- above the paper's ~0.8 instability threshold" if kl > 0.8 else ""
            print(f"  {lab:<18}{kl:>9.4f}{flag}")

        # Look the row up by label rather than zipping two filtered lists: a
        # misalignment there would silently correlate the wrong pairs.
        by_label = {lab: d for lab, d in rows}
        paired = [(kl, sum(by_label[lab]["tasks"][t]["token_acc"] for t in tasks) / len(tasks))
                  for lab, kl in kls]
        if len(paired) >= 3:
            r, p, n = clmetrics.pearson([x for x, _ in paired], [y for _, y in paired])
            ps = "n/a" if p is None else f"{p:.4f}"
            print(f"\n  KL vs mean accuracy: r={r:+.3f}  p={ps}  n={n}")
            print("  (paper: r=-0.497, p<0.001 — sign and rough magnitude is the")
            print("   gate; n is tiny in one run, so pool seeds before trusting p)")

    total = sum(d.get("seconds_total", 0) for _, d in rows)
    print(f"\n{len(rows)} boundaries, {total:.1f} s of probing in total.")


if __name__ == "__main__":
    main()
