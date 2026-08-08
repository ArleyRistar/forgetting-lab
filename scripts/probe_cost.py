"""Measure what a boundary probe actually costs (phase-1a task 3, step 4).

The plan estimated ~4 min per boundary for 8 tasks by arithmetic. Arithmetic is
not a result (CLAUDE.md hard rule 3), so this runs the real probe against the
real base model and reports the real number, plus the baseline NLL every later
boundary in a run is measured against.

Usage: uv run scripts/probe_cost.py [--batch-size 4]
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flab import probes
from flab.runconfig import RunConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/dev-3stage.yaml")
    p.add_argument("--batch-size", type=int, default=None, help="override for cost sweeps only")
    args = p.parse_args()

    cfg = RunConfig.load(args.config)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    t0 = time.perf_counter()
    # transformers 5.x renamed torch_dtype -> dtype (LAB-NOTES quirk 2)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype="bfloat16").cuda()
    load_s = time.perf_counter() - t0

    torch.cuda.reset_peak_memory_stats()
    out = probes.probe_all(
        model, tok, cfg.probe_tasks,
        n_eval=cfg.probe.n_eval, max_length=cfg.probe.max_length,
        batch_size=args.batch_size or cfg.probe.batch_size, seed=cfg.seed,
    )
    out["model_load_seconds"] = round(load_s, 1)
    out["batch_size"] = args.batch_size or cfg.probe.batch_size
    out["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    # Budget against reserved, not allocated: the allocator holds 8-13% more
    # than it hands out and reserved is what occupies the card (LAB-NOTES).
    out["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
    print("PROBECOST " + json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
