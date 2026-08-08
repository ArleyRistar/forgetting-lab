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

    total = sum(d.get("seconds_total", 0) for _, d in rows)
    print(f"\n{len(rows)} boundaries, {total:.1f} s of probing in total.")


if __name__ == "__main__":
    main()
