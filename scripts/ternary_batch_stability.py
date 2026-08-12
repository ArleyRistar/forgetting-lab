"""Characterise ternary forward-pass instability to batch composition.

Found incidentally in phase 2b: scoring the same prompts at batch 1 vs batch 8
moves ternary log-probs by 1.2-1.9 nats while the float twin moves 4e-5. Five
orders of magnitude. This measures what it actually is, because it affects every
measurement anyone takes on a ternary model — evals, KL probes, logprob
comparisons — and our literature sweeps found no report of it.

Four questions, all forward-only:

  1. how does drift scale with batch size?
  2. is it PADDING (ragged batches) or batch composition itself? -> equal-length
     prompts remove padding entirely; if drift survives, padding is not the cause
  3. which quantiser causes it — activations, weights, or both?
  4. does it change the GREEDY output, or is it confined to logprobs? A logprob
     nuisance is a measurement caveat; an argmax flip is a correctness problem
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from flab import bitlinear as bl
from flab import loading, prompts, synthetic

BASE = "HuggingFaceTB/SmolLM2-360M"
TERNARY = "outputs/phase2b/ternary-AB-s0/stage-0-synth-conflict-b"
FLOAT = "outputs/phase2b/float-AB-s0/stage-0-synth-conflict-b"
BATCHES = (1, 2, 4, 8, 16, 32)


@torch.no_grad()
def logits_at_end(model, tok, texts, batch_size, pad_free=False):
    """Log-probs at the final position. `pad_free` uses equal-length inputs so
    no padding exists at all, separating padding from batch composition."""
    dev = next(model.parameters()).device
    pad = tok.pad_token_id or 0
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    if pad_free:
        n = min(len(e) for e in enc)
        enc = [e[-n:] for e in enc]            # truncate all to the same length
    out = []
    for i in range(0, len(enc), batch_size):
        ch = enc[i:i + batch_size]
        w = max(len(c) for c in ch)
        x = torch.full((len(ch), w), pad, dtype=torch.long)
        a = torch.zeros((len(ch), w), dtype=torch.long)
        for r, c in enumerate(ch):
            x[r, w - len(c):] = torch.tensor(c)
            a[r, w - len(c):] = 1
        lg = model(input_ids=x.to(dev), attention_mask=a.to(dev)).logits
        out.append(F.log_softmax(lg[:, -1, :].float(), dim=-1))
    return torch.cat(out)


def drift(ref, cur):
    d = (ref - cur).abs()
    flips = int((ref.argmax(-1) != cur.argmax(-1)).sum())
    return {"max": float(d.max()), "median": float(d.median()),
            "mean": float(d.mean()), "argmax_flips": flips, "n": ref.shape[0]}


def scale_is_row_local(n_rows: int = 8, dim: int = 64) -> dict:
    """Is the activation scale computable from one row alone?

    The standard reason quantised inference goes batch-dependent is a scale taken
    ACROSS the batch (`x.abs().max()` over the whole tensor), which is a filed bug
    elsewhere and a security paper. Ours is `dim=-1`. Checked rather than argued,
    because "our scale cannot see other rows" is the load-bearing sentence of the
    novelty claim and the project rule is that measured beats reasoned.
    """
    torch.manual_seed(0)
    row = torch.randn(1, dim)
    alone = bl.activation_quant(row.clone())
    # ...same row, batched with rows two orders of magnitude larger
    crowd = torch.cat([row, torch.randn(n_rows - 1, dim) * 100])
    batched = bl.activation_quant(crowd)[0:1]
    return {"bit_identical": bool(torch.equal(alone, batched)),
            "max_abs_diff": float((alone - batched).abs().max()),
            "companion_scale_ratio": float(crowd[1:].abs().max() / row.abs().max())}


def main() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    keys = sorted({r["prompt"] for r in
                   synthetic.make("synth-conflict-a", "eval", n_keys=64, seed=0)})
    texts = [prompts.render("flab", "synth-conflict-a", k, "A", tok)[0] for k in keys]
    lens = sorted({len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts})
    print(f"{len(texts)} prompts, token lengths {lens[0]}-{lens[-1]}\n", flush=True)
    res = {"prompt_lengths": [lens[0], lens[-1]], "n_prompts": len(texts)}
    res["scale_is_row_local"] = scale_is_row_local()
    print(f"activation scale is row-local: {res['scale_is_row_local']}\n", flush=True)

    for label, path, ternary in (("ternary", TERNARY, True), ("float", FLOAT, False)):
        model, n = loading.load_converted(path, dtype=torch.float32,
                                          force_ternary=ternary or None)
        print(f"=== {label} ({n} BitLinears) ===", flush=True)

        # 1 + 4: drift and argmax flips vs batch size, ragged batches
        ref = logits_at_end(model, tok, texts, 1)
        res.setdefault("vs_batch", {})[label] = {}
        for b in BATCHES[1:]:
            d = drift(ref, logits_at_end(model, tok, texts, b))
            res["vs_batch"][label][b] = d
            print(f"  batch {b:>2}: max {d['max']:.3e}  median {d['median']:.3e}"
                  f"  argmax flips {d['argmax_flips']}/{d['n']}", flush=True)

        # 2: padding or batch composition?
        ref_pf = logits_at_end(model, tok, texts, 1, pad_free=True)
        d_pf = drift(ref_pf, logits_at_end(model, tok, texts, 8, pad_free=True))
        res.setdefault("pad_free", {})[label] = d_pf
        print(f"  PAD-FREE batch 8: max {d_pf['max']:.3e}  median "
              f"{d_pf['median']:.3e}  flips {d_pf['argmax_flips']}", flush=True)
        del model
        torch.cuda.empty_cache()

    # 3: which quantiser? patch one at a time on the ternary twin
    print("\n=== which quantiser causes it (ternary twin) ===", flush=True)
    res["by_quantiser"] = {}
    for name, patch in (("both (as shipped)", None),
                        ("weights only", "act"),
                        ("activations only", "wt")):
        model, _ = loading.load_converted(TERNARY, dtype=torch.float32,
                                          force_ternary=True)
        saved_a, saved_w = bl.activation_quant, bl.weight_quant
        if patch == "act":
            bl.activation_quant = lambda x: x
        elif patch == "wt":
            bl.weight_quant = lambda w: w
        try:
            ref = logits_at_end(model, tok, texts, 1)
            d = drift(ref, logits_at_end(model, tok, texts, 8))
        finally:
            bl.activation_quant, bl.weight_quant = saved_a, saved_w
        res["by_quantiser"][name] = d
        print(f"  {name:<20} max {d['max']:.3e}  median {d['median']:.3e}",
              flush=True)
        del model
        torch.cuda.empty_cache()

    Path("outputs/ternary-batch-stability.json").write_text(json.dumps(res, indent=2))
    print("\nwrote outputs/ternary-batch-stability.json")


if __name__ == "__main__":
    main()
