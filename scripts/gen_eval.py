"""Generative exact-match eval of finished runs (phase-1b open item 14).

Decides whether the gemma/llama accuracy swap is real or a teacher-forcing
artefact. Loads each run's final checkpoint, **merges the adapter**, generates
greedily on the test split and scores normalized exact match — their metric,
not ours.

Merging is not optional: an unmerged adapter costs 1.89x on generative decode
(measured 2026-08-07), and this is by far the most expensive evaluation in the
project.

Usage: uv run scripts/gen_eval.py outputs/runs/repl-llama-s33 [...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flab import generative, probes, trace
from flab.runconfig import RunConfig


def load_merged(cfg: RunConfig, checkpoint: str):
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(cfg.model, dtype="bfloat16").cuda()
    model = PeftModel.from_pretrained(base, checkpoint)
    model = model.merge_and_unload()
    # After merging there is no live adapter, so the generative-eval guard must
    # now pass. If it does not, we are about to pay the 1.89x for nothing.
    probes.ensure_no_live_adapter(model)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--n", type=int, default=None, help="cap examples per task")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    for rd in args.run_dirs:
        rd = Path(rd)
        state = json.loads((rd / "runstate.json").read_text())
        # Rebuild from the run's own provenance rather than hunting for the YAML,
        # which may have been edited since the run finished.
        cfg = RunConfig.from_dict(json.loads((rd / "run.json").read_text())["config"])

        final = state["stages"][-1]["checkpoint"]
        if not final:
            print(f"{rd.name}: no final checkpoint, skipping")
            continue

        tok = AutoTokenizer.from_pretrained(cfg.model)
        t0 = time.perf_counter()
        model = load_merged(cfg, final)
        out = {"run": rd.name, "model": cfg.model, "seed": cfg.seed,
               "checkpoint": final, "tasks": {}}

        for task in cfg.probe_tasks:
            ex, stats = trace.load_probe_examples(
                task, n_eval=args.n or 10_000, seed=cfg.seed,
                variant=cfg.trace_variant, split=cfg.eval_split)
            r = generative.evaluate(model, tok, task, ex, batch_size=args.batch_size,
                                    prompt_style=cfg.prompt_style)
            out["tasks"][task] = {"exact_match": r.exact_match, "n": r.n,
                                  "seconds": round(r.seconds, 1),
                                  "samples": r.examples}
            print(f"  {rd.name:<20} {task:<12} EM {r.exact_match:.4f}  n={r.n}  "
                  f"{r.seconds:.0f}s", flush=True)

        out["op"] = sum(v["exact_match"] for v in out["tasks"].values()) / len(out["tasks"])
        out["seconds_total"] = round(time.perf_counter() - t0, 1)
        (rd / "gen-eval.json").write_text(json.dumps(out, indent=2))
        print(f"  {rd.name:<20} OP (generative) = {out['op']:.4f}   "
              f"[{out['seconds_total']:.0f}s]", flush=True)

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
