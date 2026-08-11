"""Falsifiers for the retracted H2 effect (2026-08-11).

The retraction established that on the conflict pair both arms reach **zero**
task-A accuracy, so the ~6-nat gap is not retention. What is left is an
unexplained observation: the ternary twin is *less confidently wrong*. Two
explanations fit, and they make opposite predictions:

  RESIDUAL RETENTION  the true value v1 still holds probability mass above the
                      other non-B letters -> v1 beats the distractors
  ENTROPY FLOOR       the quantised forward simply has fatter tails, so ALL
                      non-B letters carry more mass in the ternary arm -> v1 is
                      level with the distractors, and the distractor itself is
                      also nats below the float twin's

So the discriminating measurement is not v1's absolute probability but **v1
against never-taught letters in the same distribution**. Three views, all from
the same forward pass:

  1. renormalised retention   p(v1) / (1 - p(v2))
  2. v1's rank among the 7 non-v2 letters (1 = still the best answer)
  3. distractor mass          mean p over letters that are neither v1 nor v2

Usage: uv run scripts/h2_falsifier.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flab import loading, probes, synthetic, trace

BASE = "HuggingFaceTB/SmolLM2-360M"
ROOT = Path("outputs/phase2")


def letter_token_ids(tok) -> dict[str, int]:
    """Token id for each single-letter value, encoded exactly as the probe does.

    Built by tokenising the answer string alone, which is how `probes._encode`
    treats it — deriving them any other way risks scoring a different token than
    training supervised.
    """
    ids = {}
    for v in synthetic.VALUES:
        enc = tok(v, add_special_tokens=False)["input_ids"]
        if len(enc) != 1:
            raise RuntimeError(f"value {v!r} is not a single token: {enc}")
        ids[v] = enc[0]
    return ids


@torch.no_grad()
def letter_probs(model, tok, prompts: list[str], ids: dict[str, int],
                 task: str = "synth-conflict-a",
                 batch_size: int = 4) -> list[dict[str, float]]:
    """Probability over the 8 letters at the answer position, per prompt.

    Encoded through `probes._encode`, NOT by tokenising the raw prompt. The probe
    wraps every prompt in the chat template (`<|user|>...<|assistant|>`), and a
    first version of this script fed the bare text — scoring the model on an
    input format it had never been trained on. It showed p(v2)=0.18 where both
    arms reach B-NLL ~0.0000 on those very prompts, i.e. ~1.0; that impossibility
    is what exposed it. Values are single tokens, so dropping the final token
    gives the identical prefix whichever letter is passed in.
    """
    device = next(model.parameters()).device
    pad = tok.pad_token_id or 0
    model.eval()
    order = list(synthetic.VALUES)
    cols = torch.tensor([ids[v] for v in order], device=device)
    out: list[dict[str, float]] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = []
        for q in chunk:
            full, _, _ = probes._encode(tok, q, synthetic.VALUES[0], 256,
                                        prompt_style="flab", task=task)
            enc.append(full[:-1])          # prefix up to the answer position
        width = max(len(e) for e in enc)
        # LEFT-pad so the final position is the answer position for every row.
        inp = torch.full((len(chunk), width), pad, dtype=torch.long)
        for r, e in enumerate(enc):
            inp[r, width - len(e):] = torch.tensor(e)
        inp = inp.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(input_ids=inp).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)[:, cols]
        for r in range(len(chunk)):
            out.append({v: float(probs[r, j]) for j, v in enumerate(order)})
    return out


def analyse(pv: list[dict[str, float]], v1: list[str], v2: list[str]) -> dict:
    ren, ranks, dist, p1, p2 = [], [], [], [], []
    for probs, a, b in zip(pv, v1, v2):
        others = [v for v in synthetic.VALUES if v not in (a, b)]
        p1.append(probs[a]); p2.append(probs[b])
        ren.append(probs[a] / max(1e-12, 1.0 - probs[b]))
        non_b = sorted(((probs[v], v) for v in synthetic.VALUES if v != b),
                       reverse=True)
        ranks.append(1 + [v for _, v in non_b].index(a))
        dist.append(sum(probs[v] for v in others) / len(others))
    n = len(pv)
    return {
        "n": n,
        "p_v1": sum(p1) / n, "p_v2": sum(p2) / n,
        "renormalised_retention": sum(ren) / n,
        "mean_rank_of_v1_among_7": sum(ranks) / n,
        "frac_v1_is_top_non_b": sum(1 for r in ranks if r == 1) / n,
        "mean_distractor_prob": sum(dist) / n,
        # The discriminator: retention puts v1 well above a never-taught letter;
        # an entropy floor puts them level.
        "v1_over_distractor": (sum(p1) / n) / max(1e-12, sum(dist) / n),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--bonly", action="store_true",
                   help="score the B-ONLY control instead: models that learned B "
                        "and NEVER learned A. If v1 is still elevated there, the "
                        "elevation is a baseline property, not residual memory.")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = letter_token_ids(tok)

    ex_a, _ = trace.load_probe_examples("synth-conflict-a", n_eval=50, seed=a.seed)
    b_map = {r["prompt"]: r["answer"]
             for r in synthetic.make("synth-conflict-b", "eval", seed=a.seed)}
    prompts = [e["prompt"] for e in ex_a if e["prompt"] in b_map]
    v1 = [e["answer"] for e in ex_a if e["prompt"] in b_map]
    v2 = [b_map[q] for q in prompts]
    print(f"{len(prompts)} keys with both values known "
          f"(chance p per letter = {1/synthetic.N_VALUES:.4f})\n")

    result = {"seed": a.seed, "steps": a.steps, "bonly": a.bonly,
              "n_keys": len(prompts),
              "chance_p": 1 / synthetic.N_VALUES, "arms": {}}

    for arm in ("ternary", "float"):
        run = (f"{arm}-BONLY-s{a.seed}-{a.steps}" if a.bonly
               else f"{arm}-conflict-s{a.seed}-Bshift-{a.steps}")
        ck = ROOT / run / "stage-0-synth-conflict-b"
        if not (ck / "model.safetensors").is_file():
            print(f"  {arm}: {ck} missing — retrain that cell first")
            continue
        model, n_bit = loading.load_converted(str(ck), dtype=torch.float32,
                                              force_ternary=(arm == "ternary"))
        if arm == "ternary":
            loading.assert_ternary(model)
        pv = letter_probs(model, tok, prompts, ids)
        r = analyse(pv, v1, v2)
        result["arms"][arm] = r
        print(f"  {arm} ({n_bit} BitLinears)")
        print(f"    p(v1) {r['p_v1']:.6f}   p(v2) {r['p_v2']:.6f}   "
              f"distractor {r['mean_distractor_prob']:.6f}")
        print(f"    v1/distractor ratio {r['v1_over_distractor']:.3f}   "
              f"(1.0 = no residual memory of v1)")
        print(f"    v1 rank among 7 non-B letters {r['mean_rank_of_v1_among_7']:.2f}"
              f"   top in {r['frac_v1_is_top_non_b']:.0%} of keys  (chance 14%)")
        del model
        torch.cuda.empty_cache()

    out_path = Path(a.out or ("outputs/phase2/h2-falsifier"
                              + ("-bonly" if a.bonly else "") + ".json"))
    result["condition"] = "B-only (A never taught)" if a.bonly else "A then B"
    out_path.write_text(json.dumps(result, indent=2))
    if len(result["arms"]) == 2:
        t, f = result["arms"]["ternary"], result["arms"]["float"]
        print("\n  VERDICT")
        print(f"    v1-over-distractor: ternary {t['v1_over_distractor']:.3f}, "
              f"float {f['v1_over_distractor']:.3f}")
        print("    both ~1.0  -> entropy floor, no residual retention in either arm")
        print("    ternary >> float -> residual retention specific to the ternary twin")
        print(f"    distractor mass: ternary {t['mean_distractor_prob']:.6f} vs "
              f"float {f['mean_distractor_prob']:.6f}"
              f"  (ratio {t['mean_distractor_prob']/max(1e-12,f['mean_distractor_prob']):.1f}x)")
        print("    a large distractor ratio with v1/distractor ~1 in both arms is"
              " the calibration explanation")


if __name__ == "__main__":
    main()
