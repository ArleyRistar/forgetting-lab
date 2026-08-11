"""Phase 2b runner — does a sub-behavioural trace survive overwriting?

Per (arm, seed), four runs from the converted twin:

    A   train synth-conflict-a          (the association to be overwritten)
    P   train synth-placebo-a           (same keys, a value that is neither v1 nor v2)
    AB  train synth-conflict-b from A   (the measurement)
    PB  train synth-conflict-b from P   (the matched control)

The A→B minus P→B contrast isolates "was v1 taught" at matched budget, matched
format exposure and matched total steps — which B-only did not.

Fresh output root on purpose: the generator changed (the v2 collision bump is
gone), and `COMPLETE`+weights skip logic would otherwise silently reuse
old-generator checkpoints from phase 2.
"""
from __future__ import annotations

import argparse, json, shutil, time
from pathlib import Path

import torch

from flab import loading, sequential
from flab.runconfig import ProbeConfig, RunConfig, StageConfig, TrainSpec

STEPS = 300
N_KEYS = 200
MICRO = {"ternary": (4, 4), "float": (2, 8)}
TASKS = ["synth-conflict-a", "synth-placebo-a", "synth-conflict-b"]
ROOT = Path("outputs/phase2b")


def cfg_for(arm, name, model, task, seed):
    batch, accum = MICRO[arm]
    return RunConfig(
        run_name=name, model=model, mode="full", optim="adamw_bnb_8bit",
        seed=seed, n_keys=N_KEYS, gen_seed=seed,
        stages=(StageConfig(task=task, learning_rate=1e-4, max_steps=STEPS),),
        train=TrainSpec(batch_size=batch, grad_accum=accum, max_length=256,
                        completion_only=True),
        probe=ProbeConfig(tasks=TASKS, n_eval=N_KEYS, max_length=256,
                          batch_size=4, reference_n=0))


def run_one(arm, tag, src, task, seed, tok, ternary):
    d = ROOT / tag
    weights = d / f"stage-0-{task}" / "model.safetensors"
    if (d / "COMPLETE").exists() and weights.is_file():
        print(f"  skip {tag}", flush=True)
        return str(d / f"stage-0-{task}")
    shutil.rmtree(d, ignore_errors=True)
    print(f"  {tag}: {task} from {src}", flush=True)
    model, n = loading.load_converted(src, dtype=torch.float32,
                                      force_ternary=ternary or None)
    if ternary:
        if n == 0:
            raise SystemExit(f"{src} loaded with 0 BitLinear layers")
        loading.assert_ternary(model)
    sequential.run(cfg_for(arm, tag, src, task, seed), d, model=model, tokenizer=tok)
    if ternary:
        loading.assert_ternary(model)      # still ternary after training
    del model
    torch.cuda.empty_cache()
    # Trainer keeps 2 rotating checkpoints (weights + optimizer) that nothing
    # else deletes; without this the transient footprint is several times the
    # 35 GB the card budgets.
    for c in (d / f"stage-0-{task}").parent.glob("stage-0-*/checkpoint-*"):
        shutil.rmtree(c, ignore_errors=True)
    return str(d / f"stage-0-{task}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="ternary,float")
    p.add_argument("--seeds", default="0,1,2")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Merge, do not overwrite. phase2.py had exactly this bug: `index = []` plus
    # one invocation per cell means the last invocation's file is the only one
    # left. Fixed there on 2026-08-10 and then reintroduced here by writing a new
    # script instead of copying the fixed one.
    idx_path = ROOT / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.is_file() else []

    for arm in a.arms.split(","):
        ternary = arm == "ternary"
        twin = f"outputs/convert/{arm}-360m/final"
        for seed in [int(x) for x in a.seeds.split(",")]:
            print(f"\n### {arm} seed {seed}", flush=True)
            a_ck = run_one(arm, f"{arm}-A-s{seed}", twin, "synth-conflict-a", seed, tok, ternary)
            p_ck = run_one(arm, f"{arm}-P-s{seed}", twin, "synth-placebo-a", seed, tok, ternary)
            ab = run_one(arm, f"{arm}-AB-s{seed}", a_ck, "synth-conflict-b", seed, tok, ternary)
            pb = run_one(arm, f"{arm}-PB-s{seed}", p_ck, "synth-conflict-b", seed, tok, ternary)
            index = [c for c in index
                     if not (c["arm"] == arm and c["seed"] == seed)]
            index.append({"arm": arm, "seed": seed, "AB": ab, "PB": pb,
                          "A": a_ck, "P": p_ck})
            idx_path.write_text(json.dumps(index, indent=2))

    print(f"\n{len(index)} (arm,seed) cells in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
