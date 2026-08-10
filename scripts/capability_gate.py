"""Item-20 capability gate for the converted twins (phase 1d task 4).

**Why this gate is blocking.** Arley's decision on item 31 scoped forgetting to
what the *converted* model relearned. So a probe the ternary twin cannot do at
t=0 has nothing to forget, and its flat forgetting curve would read as "no
forgetting" when it means "no capability" — a null that flatters the harness in
exactly the direction we would want to believe.

**Why the criterion is a shuffled-answer control rather than "chance".** NLL has
no chance level, and CLAUDE.md records that accuracies at this scale moved
nothing beyond ~1.1 SE in phase 0, so an accuracy-based test has severe power
problems at n=200. Instead we permute the answer assignment *within* the task, so
the answer distribution is identical and only the prompt→answer pairing is
broken. A model with task knowledge scores its true pairing better. The paired
difference has a well-defined SE, and the gate is a stated number of SEs.

The permutation is a derangement: with a plain shuffle, some items keep their own
answer and the control is contaminated toward the true condition — worst on tasks
with few distinct answers, which is exactly the multiple-choice case.

Usage: uv run scripts/capability_gate.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from flab import loading, probes, trace

BASE = "HuggingFaceTB/SmolLM2-360M"
CANDIDATES = ["FOMC", "ScienceQA", "Py150", "NumGLUE-cm",
              "synth-conflict-a", "synth-disjoint-a"]
SE_THRESHOLD = 3.0


def derangement(n: int, seed: int) -> list[int]:
    """A permutation with no fixed point, so no item keeps its own answer."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        p = torch.randperm(n, generator=g).tolist()
        if all(p[i] != i for i in range(n)):
            return p
    # Fall back to a rotation, which is a derangement by construction.
    return [(i + 1) % n for i in range(n)]


@torch.no_grad()
def answer_nll_per_item(model, tokenizer, pairs, task: str,
                        max_length: int = 1024, batch_size: int = 4) -> list[float]:
    """Mean per-token NLL over each item's answer tokens.

    Per *token*, not per item: the shuffled control pairs prompts with answers of
    different lengths, and a total-NLL comparison would then partly measure
    length rather than fit.
    """
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id or 0
    encoded = []
    for i, (prompt, answer) in enumerate(pairs):
        ids, labels, _ = probes._encode(tokenizer, prompt, answer, max_length,
                                        prompt_style="flab", task=task)
        if ids is not None:
            encoded.append((i, ids, labels))

    out: dict[int, float] = {}
    encoded.sort(key=lambda t: len(t[1]))
    model.eval()
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start:start + batch_size]
        width = max(len(ids) for _, ids, _ in chunk)
        inp = torch.full((len(chunk), width), pad, dtype=torch.long)
        lab = torch.full((len(chunk), width), -100, dtype=torch.long)
        for r, (_, ids, labels) in enumerate(chunk):
            inp[r, :len(ids)] = torch.tensor(ids)
            lab[r, :len(labels)] = torch.tensor(labels)
        inp, lab = inp.to(device), lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(input_ids=inp).logits
        lp = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            lab[:, 1:].reshape(-1), reduction="none", ignore_index=-100
        ).view(len(chunk), -1)
        counts = (lab[:, 1:] != -100).sum(dim=1)
        totals = lp.sum(dim=1)
        for r, (idx, _, _) in enumerate(chunk):
            if counts[r] > 0:
                out[idx] = float(totals[r] / counts[r])
    return [out.get(i, float("nan")) for i in range(len(pairs))]


def gate_one(model, tokenizer, task: str, n_eval: int, seed: int,
              batch_size: int = 2) -> dict:
    examples, stats = trace.load_probe_examples(task, n_eval=n_eval, seed=seed)
    pairs = [(e["prompt"], e["answer"]) for e in examples]
    if len(pairs) < 10:
        return {"task": task, "verdict": "SKIP", "reason": f"only {len(pairs)} examples"}

    perm = derangement(len(pairs), seed)
    shuffled = [(pairs[i][0], pairs[perm[i]][1]) for i in range(len(pairs))]

    true_nll = answer_nll_per_item(model, tokenizer, pairs, task, batch_size=batch_size)
    ctrl_nll = answer_nll_per_item(model, tokenizer, shuffled, task, batch_size=batch_size)

    d = [c - t for c, t in zip(ctrl_nll, true_nll)
         if not (math.isnan(c) or math.isnan(t))]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else float("inf")
    se = math.sqrt(var / n)
    n_se = mean / se if se > 0 else float("inf")
    return {
        "task": task, "n_scored": n,
        "true_nll": sum(t for t in true_nll if not math.isnan(t)) / n,
        "control_nll": sum(c for c in ctrl_nll if not math.isnan(c)) / n,
        "paired_delta": mean, "se": se, "n_se": n_se,
        "verdict": "KEEP" if n_se >= SE_THRESHOLD else "DROP",
        "examples_available": stats.get("available"),   # key is "available" (trace.py:188)
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks", default=",".join(CANDIDATES))
    p.add_argument("--batch-size", type=int, default=2,
                   help="2 by default: ScienceQA's long prompts triggered CUDA "
                        "allocator retries at 4 on the 8GB card")
    p.add_argument("--out", default="outputs/null/capability-gate.json")
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tasks = a.tasks.split(",")
    result = {"n_eval": a.n_eval, "seed": a.seed,
              "criterion": f"paired (control - true) answer NLL >= {SE_THRESHOLD} SE",
              "arms": {}}

    # `base` is a POSITIVE CONTROL for the gate itself, not a candidate twin.
    # If the untouched pretrained model also shows no discrimination, the gate is
    # broken and "the twins lost the capability" is an unsafe reading of a null.
    for arm in ("base", "ternary", "float"):
        src = BASE if arm == "base" else f"outputs/convert/{arm}-360m/final"
        print(f"\n=== {arm} ({src}) ===", flush=True)
        model, n_bit = loading.load_converted(src, dtype=torch.float32)
        if arm == "ternary" and n_bit == 0:
            raise SystemExit("ternary twin loaded with 0 BitLinear layers")
        print(f"  {n_bit} BitLinear layers", flush=True)
        rows = []
        for task in tasks:
            t0 = time.perf_counter()
            r = gate_one(model, tok, task, a.n_eval, a.seed, a.batch_size)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            rows.append(r)
            if r["verdict"] == "SKIP":
                print(f"  {task:<18} SKIP ({r['reason']})", flush=True)
            else:
                print(f"  {task:<18} true {r['true_nll']:.4f}  ctrl {r['control_nll']:.4f}"
                      f"  delta {r['paired_delta']:+.4f}  {r['n_se']:>7.1f} SE  "
                      f"{r['verdict']}", flush=True)
        result["arms"][arm] = rows
        del model
        torch.cuda.empty_cache()

    base_keep = {t["task"] for t in result["arms"]["base"] if t["verdict"] == "KEEP"}
    result["gate_instrument_ok"] = len(base_keep) > 0
    result["base_discriminates"] = sorted(base_keep)
    keep = {t["task"] for t in result["arms"]["ternary"] if t["verdict"] == "KEEP"}
    result["surviving_probes"] = sorted(keep)
    result["gate_passes"] = len(keep) > 0
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"\nbase model discriminates on: {sorted(base_keep) or 'NOTHING — THE GATE ITSELF IS BROKEN'}")
    print(f"surviving probes (ternary twin): {sorted(keep) or 'NONE — gate FAILS'}")


if __name__ == "__main__":
    main()
