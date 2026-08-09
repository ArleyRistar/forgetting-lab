"""Measure the conversion quality gap for the 1c pair (spec §6 1c).

Held-out loss for three models on the *same* held-out tokens:

  * the original SmolLM2-360M, untouched
  * the float twin, continued-trained on the conversion corpus
  * the ternary twin, converted via QAT on the same tokens

Spec §6 1c asks us to *measure and report* the gap rather than minimise it, and
§9 has phase 2 compare each twin against **its own** post-conversion baseline —
so the absolute gap is documentation, not a target.

**Disjointness is structural, not probabilistic.** An earlier version of this
script drew the held-out set with `shuffle(seed=999)` against training's
`seed=0` and claimed that made it disjoint. That was wrong: a buffered shuffle
permutes order (and shard order), it does not partition a corpus, so the two
streams draw from the same `sample-10BT` and can overlap. Instead we now use the
*same* seed-0 stream training used and `.skip()` past everything training
consumed — the generator is deterministic and single-process, so every block
after the skip is provably unseen.

Training consumed 4000 steps x 16 blocks = 64,000 blocks = 65.5M tokens. At a
measured 944.35 mean tokens/row (400-row sample, 2026-08-09) that is ~69,400
rows, so SKIP_ROWS below is a ~3.6x margin on a quantity we measured rather than
assumed.

WikiText-103 test is scored as a second, out-of-distribution held-out set: it is
cheap, it cannot overlap the training stream by construction, and the HF
1.58-bit blog reports WikiText perplexity, so it makes our numbers comparable to
the closest published run.

Both twins are scored on identical token blocks, which is the only way the
comparison means anything.

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

# Rows to skip before drawing held-out blocks. Training needed ~69,400 rows for
# its 65.5M tokens (944.35 measured mean tokens/row), so this is a ~3.6x margin.
# Raise it if the training budget ever grows.
SKIP_ROWS = 250_000
TRAIN_ROWS_ESTIMATE = 69_400


def heldout_blocks(tok, n_blocks: int, seq_len: int, seed: int, skip_rows: int):
    """Held-out blocks from the SAME stream training used, skipped past its end.

    Same seed as training on purpose: that reproduces training's exact block
    order, so skipping past the rows it consumed is a *proof* of disjointness
    rather than a hope. A different seed would only reshuffle the same corpus.

    Materialised once and reused for every model — scoring three models on
    three different samples would make the comparison meaningless.
    """
    from datasets import load_dataset

    ds = (load_dataset(DATASET, name=DATASET_CONFIG, split="train",
                       streaming=True)
          .shuffle(seed=seed, buffer_size=10_000)   # matches convert.py
          .skip(skip_rows))
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


def wikitext_blocks(tok, n_blocks: int, seq_len: int):
    """WikiText-103 test as an out-of-distribution held-out set.

    Cannot overlap the FineWeb-edu training stream by construction, and the HF
    1.58-bit blog reports WikiText perplexity, so this is the number that makes
    us comparable to the closest published run.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    buf, blocks = [], []
    for row in ds:
        if not row["text"].strip():
            continue
        buf.extend(tok(row["text"], add_special_tokens=False)["input_ids"])
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
    p.add_argument("--seed", type=int, default=0,
                   help="MUST match training's seed: we reproduce training's "
                        "block order and skip past it, rather than reshuffling")
    p.add_argument("--skip-rows", type=int, default=SKIP_ROWS)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    print(f"building {a.n_blocks} held-out blocks: seed {a.seed} (same stream as "
          f"training), skipping {a.skip_rows:,} rows past training's ~"
          f"{TRAIN_ROWS_ESTIMATE:,}", flush=True)
    corpora = {"fineweb_heldout": heldout_blocks(tok, a.n_blocks, a.seq_len,
                                                 a.seed, a.skip_rows)}
    print("building wikitext-103 test blocks (out-of-distribution)", flush=True)
    corpora["wikitext103_test"] = wikitext_blocks(tok, a.n_blocks, a.seq_len)

    out = {"n_blocks": a.n_blocks, "seq_len": a.seq_len, "heldout_seed": a.seed,
           "skip_rows": a.skip_rows, "train_rows_estimate": TRAIN_ROWS_ESTIMATE,
           "disjointness": "structural: same stream as training, skipped past it",
           "corpora": {}}

    # Load each model ONCE and score it on every corpus — loading a 360M model
    # three times per corpus would triple the run for no reason.
    scores: dict[str, dict[str, float]] = {}
    for kind, path in (("base", None), ("float", a.float_), ("ternary", a.ternary)):
        if path and not Path(path).exists():
            print(f"  {kind}: {path} missing, skipping", flush=True)
            continue
        t0 = time.perf_counter()
        model = load(kind, path)
        for cname, blocks in corpora.items():
            loss, n_tok = heldout_loss(model, blocks)
            scores.setdefault(cname, {})[kind] = loss
            out["corpora"].setdefault(cname, {})[kind] = {
                "loss": loss, "ppl": float(torch.exp(torch.tensor(loss))),
                "n_tokens": n_tok}
            print(f"  {kind:<8} {cname:<18} loss {loss:.4f}  "
                  f"ppl {float(torch.exp(torch.tensor(loss))):.2f}", flush=True)
        out["corpora"]["seconds_" + kind] = round(time.perf_counter() - t0, 1)
        del model
        torch.cuda.empty_cache()

    out["gaps"] = {}
    for cname, s in scores.items():
        g = {}
        if "base" in s and "ternary" in s:
            g["ternary_vs_base"] = s["ternary"] - s["base"]
        if "float" in s and "ternary" in s:
            # The comparison that matters: both twins saw the same tokens, so
            # this isolates ternarisation from the cost of the extra training.
            g["ternary_vs_float_twin"] = s["ternary"] - s["float"]
        if "base" in s and "float" in s:
            g["float_twin_vs_base"] = s["float"] - s["base"]
        out["gaps"][cname] = g

    Path("outputs/convert/conversion-gap.json").write_text(json.dumps(out, indent=2))
    print("GAP " + json.dumps(out["gaps"]))


if __name__ == "__main__":
    main()
