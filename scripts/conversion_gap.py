"""Measure the conversion quality gap for the 1c pair (spec §6 1c).

Held-out loss for three models on the *same* held-out tokens:

  * the original SmolLM2-360M, untouched
  * the float twin, continued-trained on the conversion corpus
  * the ternary twin, converted via QAT on the same tokens

Spec §6 1c asks us to *measure and report* the gap rather than minimise it, and
§9 has phase 2 compare each twin against **its own** post-conversion baseline —
so the absolute gap is documentation, not a target.

The held-out split is FineWeb-edu with a different seed from training, so it is
disjoint from what either twin saw. Both twins are scored on identical token
blocks, which is the only way the comparison means anything.

Usage: uv run scripts/conversion_gap.py
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flab import bitlinear as bl
from flab.convert import DATASET, DATASET_CONFIG

BASE = "HuggingFaceTB/SmolLM2-360M"


def heldout_blocks(tok, n_blocks: int, seq_len: int, seed: int):
    """Fixed held-out blocks, disjoint from training by seed.

    Materialised once and reused for every model — scoring three models on
    three different samples would make the comparison meaningless.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train",
                      streaming=True).shuffle(seed=seed, buffer_size=2000)
    blocks, buf = [], []
    for row in ds:
        buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len and len(blocks) < n_blocks:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(blocks) >= n_blocks:
            break
    return blocks


@torch.no_grad()
def heldout_loss(model, blocks, batch_size: int = 2) -> tuple[float, int]:
    device = next(model.parameters()).device
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        ids = torch.tensor(blocks[i : i + batch_size], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids).logits
        lp = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1), reduction="sum")
        total += lp.item()
        n += ids[:, 1:].numel()
    return total / n, n


def load(kind: str, path: str | None):
    if kind == "base":
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    elif kind == "float":
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    elif kind == "ternary":
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        # The checkpoint holds LATENT weights; ternarisation happens in the
        # forward pass, so it must be re-applied at λ=1 to score the model that
        # actually exists. Loading it without converting would score a float
        # model and report it as ternary.
        m, n = bl.convert(m, lambda_=1.0)
        print(f"  re-applied BitLinear to {n} layers at lambda=1", flush=True)
    else:
        raise ValueError(kind)
    return m.cuda() if torch.cuda.is_available() else m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ternary", default="outputs/convert/ternary-360m/final")
    p.add_argument("--float", dest="float_", default="outputs/convert/float-360m/final")
    p.add_argument("--n-blocks", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=999, help="held-out seed, != training seed 0")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    print(f"building {a.n_blocks} held-out blocks (seed {a.seed}, disjoint from training)",
          flush=True)
    blocks = heldout_blocks(tok, a.n_blocks, a.seq_len, a.seed)

    out = {"n_blocks": len(blocks), "seq_len": a.seq_len, "heldout_seed": a.seed,
           "models": {}}
    for kind, path in (("base", None), ("float", a.float_), ("ternary", a.ternary)):
        if path and not Path(path).exists():
            print(f"  {kind}: {path} missing, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        model = load(kind, path)
        loss, n_tok = heldout_loss(model, blocks)
        out["models"][kind] = {"loss": loss, "ppl": float(torch.exp(torch.tensor(loss))),
                               "n_tokens": n_tok, "seconds": round(time.perf_counter() - t0, 1)}
        print(f"  {kind:<8} loss {loss:.4f}  ppl {out['models'][kind]['ppl']:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    m = out["models"]
    if "base" in m and "ternary" in m:
        out["gap_ternary_vs_base"] = m["ternary"]["loss"] - m["base"]["loss"]
    if "float" in m and "ternary" in m:
        # The comparison that matters: both twins saw the same tokens, so this
        # isolates the cost of ternarisation from the cost of the extra training.
        out["gap_ternary_vs_float_twin"] = m["ternary"]["loss"] - m["float"]["loss"]
    if "base" in m and "float" in m:
        out["float_twin_vs_base"] = m["float"]["loss"] - m["base"]["loss"]

    Path("outputs/convert/conversion-gap.json").write_text(json.dumps(out, indent=2))
    print("GAP " + json.dumps({k: v for k, v in out.items() if k != "models"}))


if __name__ == "__main__":
    main()
