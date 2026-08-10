"""Background flip-rate null arm (phase 1d task 3, open item 24).

Continues a converted twin on FineWeb-edu with **no distribution shift**, densely
checkpointed, so phase 2 can report task-induced flips over this floor. Directly
analogous to the phase-1b null control that established the harness adds ~0 noise.

**Why this is not `convert.py --mode ternary`.** That path installs
`LambdaWarmup`, which pushes λ from 0 (`convert.py:71` → `bitlinear.py:160`). The
arm would spend its opening steps as a partially-float model, every early "flip"
would be conversion artefact rather than continued-training floor, and the loss
curve would look perfectly healthy throughout. `convert.py` also has no stream
skip, so the arm would re-train on tokens the conversion already saw. Both
failures are silent, which is this project's documented failure shape.

So this script instead:

* loads through `flab.loading.load_converted` (λ=1, fp32) and **asserts λ==1 every
  step** rather than scheduling it;
* runs `assert_ternary` before and after training, so a run that silently
  de-ternarised cannot be written up as ternary;
* skips past everything the conversion and the held-out eval consumed, and
  **verifies the first block actually differs** from training's first block;
* saves **weights only** — the flip analysis never reads optimizer state, and
  including it would waste 42% of every checkpoint.

Note the optimizer starts cold: `final/` contains no `optimizer.pt` (verified),
so Adam state cannot be resumed and `warmup_steps=100` applies.

**Where to put the per-step burst — both ends of a cosine are wrong.** A burst in
the first steps measures LR warmup (1e-6–1e-5) and reports a suppressed rate. A
burst at the end measures the cosine tail, which is worse: measured on the 400-step
arm, per-step flips fell 109 -> 83 -> ... -> 5 -> 0 across steps 390-400, the last
step having literally none because the LR had decayed to ~0. A cosine schedule has
no flat region, so *any* rate read off it is a rate at an unstated learning rate.
Use `--lr-scheduler constant` with the burst after `warmup_steps` for a number
quotable as "per-step flips at lr X" — which is what both the DQT comparison and
phase 2's floor actually need.

Usage:
  uv run scripts/null_arm.py --arm ternary
  uv run scripts/null_arm.py --arm float
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import (AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

from flab import bitlinear as bl
from flab import loading
from flab.convert import DATASET, DATASET_CONFIG

BASE = "HuggingFaceTB/SmolLM2-360M"
# Past training's ~69,400 rows AND the held-out set's 250,000 window, so the null
# arm's tokens are disjoint from both by construction.
SKIP_ROWS = 300_000


def stream(tokenizer, seq_len: int, seed: int, skip_rows: int):
    from datasets import IterableDataset, load_dataset

    ds = (load_dataset(DATASET, name=DATASET_CONFIG, split="train",
                       streaming=True)
          .shuffle(seed=seed, buffer_size=10_000)
          .skip(skip_rows))

    def gen():
        buf: list[int] = []
        for row in ds:
            buf.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            buf.append(tokenizer.eos_token_id)
            while len(buf) >= seq_len:
                block = buf[:seq_len]
                buf = buf[seq_len:]
                yield {"input_ids": block, "labels": block}

    return IterableDataset.from_generator(gen)


def verify_skip_is_disjoint(tokenizer, seq_len: int, seed: int, skip_rows: int) -> None:
    """The skip is the whole disjointness argument, so check it rather than
    trust it: the first block here must differ from the first block training saw."""
    from datasets import load_dataset

    def first_block(skip: int) -> list[int]:
        ds = (load_dataset(DATASET, name=DATASET_CONFIG, split="train",
                           streaming=True)
              .shuffle(seed=seed, buffer_size=10_000))
        if skip:
            ds = ds.skip(skip)
        buf: list[int] = []
        for row in ds:
            buf.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            buf.append(tokenizer.eos_token_id)
            if len(buf) >= seq_len:
                return buf[:seq_len]
        raise RuntimeError("stream exhausted")

    if first_block(skip_rows) == first_block(0):
        raise RuntimeError(
            f"skip({skip_rows}) yields the same first block as training — the "
            "null arm would re-train on seen tokens")
    print(f"  skip({skip_rows:,}) verified disjoint from training's stream", flush=True)


class AssertLambdaOne(TrainerCallback):
    """λ is a constant here, not a schedule. Asserted, not assumed."""

    def __init__(self, model):
        self.n = sum(1 for m in model.modules() if isinstance(m, bl.BitLinear))
        self.model = model

    def on_step_begin(self, args, state, control, **kw):
        bad = [n for n, m in self.model.named_modules()
               if isinstance(m, bl.BitLinear) and m.lambda_ != 1.0]
        if bad:
            raise RuntimeError(
                f"step {state.global_step}: {len(bad)} BitLinears have λ != 1 "
                f"(first: {bad[0]}) — this is not a ternary run")


class DenseSaves(TrainerCallback):
    """Save every `every` steps, and every step inside `burst`.

    Place the burst where the LR is known and flat — see the module docstring:
    on a cosine schedule neither end qualifies.
    """

    def __init__(self, every: int, burst: range):
        self.every, self.burst = every, burst
        self.saved: list[int] = []

    def on_step_end(self, args, state, control, **kw):
        step = state.global_step
        if step in self.burst or step % self.every == 0:
            control.should_save = True
            self.saved.append(step)
        return control


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("ternary", "float"), required=True)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--burst-from", type=int, default=390)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--skip-rows", type=int, default=SKIP_ROWS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optim", default="adamw_bnb_8bit")
    # A cosine schedule has no flat region: LR ramps to a peak at `warmup_steps`
    # then decays to ~0. A per-step flip rate measured anywhere on it is a rate
    # at an unstated LR. `--lr-scheduler constant` gives a short probe whose
    # answer is quotable as "per-step flips at lr X", which is what both the
    # DQT comparison and phase 2's floor actually need.
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()

    # Each arm keeps ITS OWN conversion micro-batch: the float twin needs 2x8 to
    # fit (autocast caches bf16 copies of leaf parameters, which the ternary path
    # bypasses). tokens/step is 16384 either way, so the arms stay comparable.
    batch, accum = (4, 4) if a.arm == "ternary" else (2, 8)
    src = f"outputs/convert/{a.arm}-360m/final"
    out = Path(a.output_dir or f"outputs/null/{a.arm}-360m")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading {src} through flab.loading", flush=True)
    model, n_bit = loading.load_converted(src, dtype=torch.float32)
    if a.arm == "ternary":
        if n_bit == 0:
            raise SystemExit(f"{src} loaded with 0 BitLinear layers — refusing to "
                             "run a FLOAT model and call it ternary")
        checked = loading.assert_ternary(model)
        print(f"  {n_bit} BitLinear layers, {checked} verified three-valued at λ=1",
              flush=True)
    elif n_bit != 0:
        raise SystemExit(f"{src} is the float arm but loaded {n_bit} BitLinears")

    verify_skip_is_disjoint(tok, a.seq_len, a.seed, a.skip_rows)

    callbacks = [DenseSaves(a.save_every, range(a.burst_from, a.max_steps + 1))]
    if a.arm == "ternary":
        callbacks.append(AssertLambdaOne(model))

    args = TrainingArguments(
        output_dir=str(out), max_steps=a.max_steps,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
        learning_rate=a.lr, lr_scheduler_type=a.lr_scheduler,
        adam_beta2=0.95, weight_decay=0.0, max_grad_norm=1.0, optim=a.optim,
        warmup_steps=100, bf16=torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available(),
        logging_steps=25,
        save_strategy="no",          # DenseSaves drives saving instead
        save_only_model=True,        # flips never read optimizer state
        save_total_limit=None, report_to=[], seed=a.seed,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=stream(tok, a.seq_len, a.seed, a.skip_rows),
                      callbacks=callbacks)

    t0 = time.perf_counter()
    trainer.train()
    trainer.save_model(str(out / "final"))

    if a.arm == "ternary":
        loading.assert_ternary(model)   # still ternary at the end, not just the start

    meta = {
        "arm": a.arm, "source": src, "max_steps": a.max_steps,
        "batch_size": batch, "grad_accum": accum,
        "tokens_per_step": batch * accum * a.seq_len,
        "tokens_seen": batch * accum * a.seq_len * a.max_steps,
        "lr": a.lr, "lr_scheduler": a.lr_scheduler, "optim": a.optim, "seed": a.seed,
        "skip_rows": a.skip_rows, "n_bitlinear": n_bit,
        "saved_steps": sorted(set(callbacks[0].saved)),
        "save_only_model": True,
        "lambda_constant_1": a.arm == "ternary",
        "fresh_optimizer": True, "warmup_steps": 100,
        "burst": [a.burst_from, a.max_steps],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out / "null_arm.json").write_text(json.dumps(meta, indent=2))
    print("NULL_ARM " + json.dumps({k: v for k, v in meta.items()
                                    if k != "saved_steps"}))


if __name__ == "__main__":
    main()
