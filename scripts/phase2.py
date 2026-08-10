"""Phase 2 — does flip fraction predict forgetting? (approved card 2026-08-10)

Design: train A once per (arm, pair, seed), then **branch** to three B-budgets
from that checkpoint. Branching is what makes 3 budgets affordable — retraining A
per budget would triple the largest cost for nothing.

Three B-conditions per A-checkpoint:

  conflict   B shares A's key namespace -> learning B MUST destroy A
  disjoint   B shares nothing           -> zero forgetting is the known answer
  same       B *is* A again             -> the NULL FLOOR

The `same` condition replaces the card's FineWeb-based floor re-validation and is
strictly better matched: it shares the arm, task, optimizer, schedule, cadence and
budget with the real B-runs, and differs *only* in that no new information
arrives. Flips there are what continued optimisation produces by itself, which is
exactly what phase-2 flips must be reported over.

Forgetting of A is measured against each twin's own post-A checkpoint, per item
31 — never against the base model or the converted twin.

Usage:
  uv run scripts/phase2.py --seeds 0            # pilot
  uv run scripts/phase2.py --seeds 1,2          # the rest, after the pilot is clean
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from flab import loading, sequential
from flab.runconfig import ProbeConfig, RunConfig, StageConfig, TrainSpec

A_STEPS = 300
BUDGETS = (25, 100, 300)
PAIRS = {"conflict": ("synth-conflict-a", "synth-conflict-b"),
         "disjoint": ("synth-disjoint-a", "synth-disjoint-b")}
N_EVAL = 50
ROOT = Path("outputs/phase2")


# Each arm keeps ITS OWN conversion micro-batch. The float arm needs 2x8: under
# autocast it passes leaf parameters to F.linear, so a bf16 copy of every weight
# is cached, while the ternary path passes a computed STE tensor and bypasses the
# cache entirely (measured 2026-08-09: float OOMs where ternary fits at 6774 MiB).
# tokens/step is 16384 either way, so the arms stay comparable. The first phase-2
# attempt used 4x4 for both and OOM'd on the 8th float run.
MICRO = {"ternary": (4, 4), "float": (2, 8)}


def cfg_for(arm: str, run_name: str, model: str, task: str, steps: int,
            probe_tasks: list[str], seed: int, save_steps: int | None) -> RunConfig:
    batch, accum = MICRO[arm]
    return RunConfig(
        run_name=run_name, model=model, mode="full", optim="adamw_bnb_8bit",
        seed=seed,
        stages=(StageConfig(task=task, learning_rate=1e-4, max_steps=steps),),
        train=TrainSpec(batch_size=batch, grad_accum=accum, max_length=256,
                        completion_only=True),
        probe=ProbeConfig(tasks=probe_tasks, n_eval=N_EVAL, max_length=256,
                          batch_size=4, reference_n=0),
    )


def load_arm(path: str, ternary: bool):
    model, n = loading.load_converted(path, dtype=torch.float32,
                                      force_ternary=ternary or None)
    if ternary:
        if n == 0:
            raise SystemExit(f"{path} loaded with 0 BitLinear layers")
        loading.assert_ternary(model)
    elif n != 0:
        raise SystemExit(f"{path} is the float arm but loaded {n} BitLinears")
    return model, n


def probe_value(run_dir: Path, fname: str, task: str) -> float | None:
    f = run_dir / fname
    if not f.is_file():
        return None
    d = json.loads(f.read_text())["tasks"].get(task)
    return d["nll"] if d else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0")
    p.add_argument("--arms", default="ternary,float")
    p.add_argument("--pairs", default="conflict,disjoint")
    p.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    p.add_argument("--a-steps", type=int, default=A_STEPS)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    seeds = [int(x) for x in a.seeds.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]

    # Merge with any previous invocation rather than overwriting. The first run
    # started `rows = []` and clobbered seed 0's results when seeds 1,2 launched;
    # only an incidental choice in phase2_flips.py (it copies `forgetting` into
    # each predictor row and merges by key) kept that data alive.
    results_path = ROOT / "results.json"
    rows = []
    if results_path.is_file():
        rows = [r for r in json.loads(results_path.read_text())
                if r["seed"] not in seeds]     # re-running a seed replaces it
    t_start = time.perf_counter()

    for arm in a.arms.split(","):
        ternary = arm == "ternary"
        twin = f"outputs/convert/{arm}-360m/final"
        for pair in a.pairs.split(","):
            task_a, task_b = PAIRS[pair]
            for seed in seeds:
                # ---- stage A: train once, branch from it ----------------
                a_dir = ROOT / f"{arm}-{pair}-s{seed}-A"
                if not (a_dir / "COMPLETE").exists():
                    print(f"\n### A: {arm} {pair} seed {seed}", flush=True)
                    model, _ = load_arm(twin, ternary)
                    sequential.run(cfg_for(arm, f"{arm}-{pair}-s{seed}-A", twin,
                                           task_a, a.a_steps, [task_a, task_b],
                                           seed, None), a_dir, model=model,
                                   tokenizer=tok)
                    if ternary:
                        loading.assert_ternary(model)   # still ternary after A
                    del model
                    torch.cuda.empty_cache()
                a_ckpt = str(a_dir / f"stage-0-{task_a}")
                nll_a_at_a = probe_value(a_dir, "probe-after-0.json", task_a)

                # ---- branch: three B-conditions x three budgets ---------
                for cond, tb in (("shift", task_b), ("same", task_a)):
                    for steps in budgets:
                        tag = f"{arm}-{pair}-s{seed}-B{cond}-{steps}"
                        b_dir = ROOT / tag
                        if (b_dir / "COMPLETE").exists():
                            print(f"  skip {tag} (done)", flush=True)
                        else:
                            print(f"  B[{cond}] {steps} steps: {tag}", flush=True)
                            model, _ = load_arm(a_ckpt, ternary)
                            sequential.run(
                                cfg_for(arm, tag, a_ckpt, tb, steps,
                                        [task_a, task_b], seed, 25),
                                b_dir, model=model, tokenizer=tok)
                            if ternary:
                                loading.assert_ternary(model)
                            del model
                            torch.cuda.empty_cache()
                        nll_a_after = probe_value(b_dir, "probe-after-0.json", task_a)
                        rows.append({
                            "arm": arm, "pair": pair, "seed": seed,
                            "condition": cond, "b_steps": steps,
                            "task_a": task_a, "task_b": tb,
                            "nll_a_at_A": nll_a_at_a,
                            "nll_a_after_B": nll_a_after,
                            "forgetting": (None if (nll_a_at_a is None or nll_a_after is None)
                                           else nll_a_after - nll_a_at_a),
                            "nll_b_after_B": probe_value(b_dir, "probe-after-0.json", tb),
                            "a_ckpt": a_ckpt, "b_ckpt": str(b_dir / f"stage-0-{tb}"),
                        })
                        ROOT.mkdir(parents=True, exist_ok=True)
                        (ROOT / "results.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{len(rows)} rows in {(time.perf_counter()-t_start)/60:.1f} min")
    for r in rows:
        f = r["forgetting"]
        print(f"  {r['arm']:<8}{r['pair']:<9}s{r['seed']} {r['condition']:<6}"
              f"{r['b_steps']:>4}  forgetting {f:+.4f}" if f is not None else "  (missing)")


if __name__ == "__main__":
    main()
