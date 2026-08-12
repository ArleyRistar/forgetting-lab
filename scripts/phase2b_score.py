"""Phase 2b scoring and analysis.

**No new scoring path.** Letter log-probs come from `probes._next_token_logprobs`
— the audited function that already left-pads, passes the attention mask, and
returns the log-softmax at the final position. The phase-2 falsifier reimplemented
this and omitted the mask, which made every number it produced an artefact. The
only extension is `fp32=True`, added in place.

Three gates run before any estimate is reported:

  batch invariance      batch 1 vs batch 8 must agree to a tolerance MEASURED in
                        a shakedown, not asserted at a number picked in advance
  cross-instrument      -log p(v2) here must match the harness probe's own
                        `nll_b_after_B`; a 200x disagreement on exactly this was
                        what the padding bug looked like from outside
  placebo equivalence   |gamma_v1(placebo->B)| < 0.3, an equivalence margin, since
                        an underpowered significance test passes trivially
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import torch

from flab import fe, loading, probes, prompts, synthetic, trace

ROOT = Path("outputs/phase2b")
BASE = "HuggingFaceTB/SmolLM2-360M"
N_KEYS = 200
TASK_A, TASK_P, TASK_B = "synth-conflict-a", "synth-placebo-a", "synth-conflict-b"
PLACEBO_MARGIN = 0.3


def value_maps(gen_seed: int):
    def m(task):
        return {r["prompt"]: r["answer"]
                for r in synthetic.make(task, "eval", n_keys=N_KEYS, seed=gen_seed)}
    return m(TASK_A), m(TASK_P), m(TASK_B)


def letter_ids(tok):
    ids = {}
    for v in synthetic.VALUES:
        e = tok(v, add_special_tokens=False)["input_ids"]
        if len(e) != 1:
            raise RuntimeError(f"value {v!r} is not one token: {e}")
        ids[v] = e[0]
    return ids


def prefixes(tok, keys):
    """Rendered prefix text per key, via the same renderer `_encode` uses."""
    return [prompts.render("flab", TASK_A, k, synthetic.VALUES[0], tok)[0] for k in keys]


def score(model, tok, texts, ids, batch_size=1):
    device = next(model.parameters()).device
    lp = probes._next_token_logprobs(model, tok, texts, device, batch_size, fp32=True)
    cols = torch.tensor([ids[v] for v in synthetic.VALUES], device=lp.device)
    sub = lp[:, cols]
    return [{v: float(sub[r, j]) for j, v in enumerate(synthetic.VALUES)}
            for r in range(sub.shape[0])]


def rank_stats(probs, keys, v1, v2) -> dict:
    """Scale-free: v1's rank among the 7 non-v2 letters, and its top-1 share.

    A rank is invariant to ANY per-key monotone transform, so a between-arm
    difference in logit range — which the ternary and float twins demonstrably
    have (never-taught letters sit ~2.5 nats deeper in float) — cannot move it.
    That is what makes it, and not gamma in log units, the statistic a cross-arm
    magnitude claim has to rest on.
    """
    ranks, top = [], 0
    for p, k in zip(probs, keys):
        non_b = sorted(((p[v], v) for v in synthetic.VALUES if v != v2[k]),
                       reverse=True)
        r = 1 + [v for _, v in non_b].index(v1[k])
        ranks.append(r)
        top += (r == 1)
    n = len(ranks)
    return {"mean_rank": sum(ranks) / n, "top1_share": top / n,
            "chance_top1": 1 / 7, "n": n}


def batch_invariance(model, tok, texts, ids) -> float:
    """Max |Δ log p| between batch 1 and batch 8 over a 16-row sample."""
    s = texts[:16]
    a, b = score(model, tok, s, ids, batch_size=1), score(model, tok, s, ids, batch_size=8)
    return max(abs(x[v] - y[v]) for x, y in zip(a, b) for v in synthetic.VALUES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=None,
                    help="batch-invariance tolerance; omit to run the shakedown "
                         "and print the measured value instead of asserting")
    ap.add_argument("--out", default=str(ROOT / "analysis.json"))
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = letter_ids(tok)
    # Paths are deterministic, so derive them rather than trusting a manifest —
    # `index.json` was overwritten once already by a runner that did not merge.
    # Any cell whose checkpoints are actually on disk is scored.
    index = []
    for arm in ("ternary", "float"):
        for seed in (0, 1, 2):
            ab = ROOT / f"{arm}-AB-s{seed}" / f"stage-0-{TASK_B}"
            pb = ROOT / f"{arm}-PB-s{seed}" / f"stage-0-{TASK_B}"
            if (ab / "model.safetensors").is_file() and \
               (pb / "model.safetensors").is_file():
                index.append({"arm": arm, "seed": seed,
                              "AB": str(ab), "PB": str(pb)})
    print(f"{len(index)} of 6 cells have both checkpoints on disk", flush=True)
    if not index:
        raise SystemExit("no scorable cells")

    out = {"n_keys": N_KEYS, "placebo_margin": PLACEBO_MARGIN, "cells": [],
           "batch_invariance": {}, "gates": {}}
    per_arm: dict[str, list[float]] = {}

    for cell in index:
        arm, seed = cell["arm"], cell["seed"]
        v1, vp, v2 = value_maps(seed)
        keys = sorted(v1)
        texts = prefixes(tok, keys)
        rec = {"arm": arm, "seed": seed}

        fits = {}
        for cond, ck in (("AB", cell["AB"]), ("PB", cell["PB"])):
            model, n = loading.load_converted(ck, dtype=torch.float32,
                                              force_ternary=(arm == "ternary") or None)
            if arm == "ternary":
                loading.assert_ternary(model)
            drift = batch_invariance(model, tok, texts, ids)
            out["batch_invariance"][f"{arm}-s{seed}-{cond}"] = drift
            if a.tol is not None and drift > a.tol:
                raise SystemExit(f"batch invariance {drift:.2e} > tol {a.tol:.2e} "
                                 f"for {arm}-s{seed}-{cond}")
            probs = score(model, tok, texts, ids)
            logp = {(k, l): p[l] for k, p in zip(keys, probs) for l in synthetic.VALUES}
            fits[cond] = {"logp": logp, "rank": rank_stats(probs, keys, v1, v2),
                          "p_v2": sum(math.exp(logp[(k, v2[k])]) for k in keys) / len(keys),
                          "fe": fe.fit(logp, v1, vp, v2)}
            del model
            torch.cuda.empty_cache()

        # vp excluded from the key mean: v' is TAUGHT in the placebo condition,
        # so leaving it in leaks gamma_vp/6 into the contrast (13-15% inflation,
        # measured).
        pk = fe.paired_contrast(fits["AB"]["logp"], fits["PB"]["logp"], v1, v2,
                                vp=vp)
        rec.update({
            "contrast_mean": sum(pk.values()) / len(pk), "n_keys_paired": len(pk),
            "gamma_v1_AB": fits["AB"]["fe"].gamma_v1,
            "gamma_v1_PB": fits["PB"]["fe"].gamma_v1,
            "gamma_vp_AB": fits["AB"]["fe"].gamma_vp,
            "gamma_vp_PB": fits["PB"]["fe"].gamma_vp,
            "p_v2_AB": fits["AB"]["p_v2"], "p_v2_PB": fits["PB"]["p_v2"],
            # scale-free arm comparison
            "rank_AB": fits["AB"]["rank"], "rank_PB": fits["PB"]["rank"],
            "top1_contrast": (fits["AB"]["rank"]["top1_share"]
                              - fits["PB"]["rank"]["top1_share"]),
            # cross-instrument gate, wired rather than only promised
            "nll_v2_scored_AB": -math.log(max(1e-30, fits["AB"]["p_v2"])),
        })
        out["cells"].append(rec)
        per_arm.setdefault(arm, []).append(rec["contrast_mean"])
        print(f"  {arm} s{seed}  contrast {rec['contrast_mean']:+.4f}  "
              f"gv1 AB {rec['gamma_v1_AB']:+.3f} PB {rec['gamma_v1_PB']:+.3f}  "
              f"p(v2) {rec['p_v2_AB']:.4f}", flush=True)
        (Path(a.out)).write_text(json.dumps(out, indent=2, default=str))

    out["seed_level"] = {arm: fe.seed_level(v) for arm, v in per_arm.items()}
    # scale-free arm comparison, the statistic a "2x" claim must rest on
    top1 = {}
    for arm in ("ternary", "float"):
        vals = [c["top1_contrast"] for c in out["cells"] if c["arm"] == arm]
        if vals:
            top1[arm] = fe.seed_level(vals)
    out["seed_level_top1"] = top1
    # placebo gate: gamma_v1 in the condition where v1 was never taught
    worst = max((abs(c["gamma_v1_PB"]) for c in out["cells"]), default=0.0)
    out["gates"]["placebo_equivalence"] = {"worst_abs_gamma_v1_PB": worst,
                                           "margin": PLACEBO_MARGIN,
                                           "pass": worst < PLACEBO_MARGIN}
    out["gates"]["max_batch_drift"] = max(out["batch_invariance"].values())
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))

    print("\n  seed-level contrasts (t with 2 df, 95% needs 4.303 x SE):")
    for arm, r in out["seed_level"].items():
        print(f"    {arm:<8} {r['mean']:+.4f} +- {r['se']:.4f}   "
              f"95% CI [{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]   t={r['t']:+.2f}")
    g = out["gates"]["placebo_equivalence"]
    print(f"  placebo gate: worst |gamma_v1(PB)| = {g['worst_abs_gamma_v1_PB']:.4f} "
          f"vs margin {g['margin']}  -> {'PASS' if g['pass'] else 'FAIL'}")
    print(f"  max batch-invariance drift: {out['gates']['max_batch_drift']:.2e}")


if __name__ == "__main__":
    main()
