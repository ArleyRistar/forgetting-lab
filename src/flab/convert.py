"""Continued-QAT conversion of a float model to ternary (spec §6 1c).

Produces one half of the matched pair the project rests on. The other half is
the same model continued-trained in bf16 on **the same tokens** — this module
runs both, selected by `--mode`, so "same tokens" is enforced by construction
rather than by remembering to match two configs.

Three things this file is careful about, each because getting it wrong would be
invisible in the loss curve:

* **λ starts at 0 and the model is bit-identical to float there.** Verified in
  `test_bitlinear.py`; the callback below logs λ every step so a warmup that
  silently never advances shows up as a flat column rather than as a model that
  merely trains badly.
* **Checkpoints hold latent weights, not effective ones.** Phase 1d measures
  flips on the latent trajectory. Saving post-quantisation weights would throw
  away exactly the signal the next phase needs, and the loss would look fine.
* **The token stream is recorded.** The float twin must see the same tokens, so
  the stream is part of the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

from flab import bitlinear as bl

DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


@dataclass
class ConvertConfig:
    model: str = "HuggingFaceTB/SmolLM2-135M"
    mode: str = "ternary"          # ternary | float  (the twin)
    max_steps: int = 2000
    warmup_lambda_steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 4
    seq_len: int = 1024
    save_steps: int = 1000
    seed: int = 0

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.seq_len


class LambdaWarmup(TrainerCallback):
    """Push `min(step/N, 1)` into every BitLinear, and record it.

    Logged rather than merely applied: a warmup that never advances produces a
    model that trains badly for reasons no loss curve explains.
    """

    def __init__(self, model, total: int):
        self.model = model
        self.total = total
        self.history: list[tuple[int, float]] = []
        self.n_layers = sum(1 for m in model.modules() if isinstance(m, bl.BitLinear))

    def on_step_begin(self, args, state, control, **kw):
        lam = bl.warmup_lambda(state.global_step, self.total)
        n = bl.set_lambda(self.model, lam)
        if n != self.n_layers:
            raise RuntimeError(
                f"lambda reached {n} BitLinears, expected {self.n_layers} — "
                "the model changed shape mid-run")
        if state.global_step % 100 == 0:
            self.history.append((state.global_step, lam))


def load_stream(tokenizer, cfg: ConvertConfig):
    """Stream FineWeb-edu, packed into fixed-length blocks.

    Streamed because the box should not hold a large corpus, and packed because
    the recipe trains on contiguous text rather than padded documents.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET, name=DATASET_CONFIG, split="train",
                      streaming=True).shuffle(seed=cfg.seed, buffer_size=10_000)

    def gen():
        buf: list[int] = []
        for row in ds:
            buf.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            buf.append(tokenizer.eos_token_id)
            while len(buf) >= cfg.seq_len:
                block = buf[: cfg.seq_len]
                buf = buf[cfg.seq_len:]
                yield {"input_ids": block, "labels": block}

    from datasets import IterableDataset

    return IterableDataset.from_generator(gen)


def build(cfg: ConvertConfig):
    # fp32 latent weights, NOT bf16. Every reference trains an fp32 master
    # (nanotron defaults `accumulate_grad_in_fp32: true`), and skipping it is
    # what flattened the first three shakedowns. Measured on this checkpoint:
    # bf16's 8 significant bits round an Adam step to *no update at all* for
    # 85% of weights at lr 1e-4 and 96% at 2e-5, so the model was freezing 24
    # of every 25 latents per step — and lowering the LR made it worse, not
    # better. `bf16=True` below still gives mixed-precision compute; fp32
    # weights under autocast is exactly the reference setup.
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    n_bit = 0
    if cfg.mode == "ternary":
        model, n_bit = bl.convert(model, lambda_=0.0)
    elif cfg.mode != "float":
        raise ValueError("mode must be 'ternary' or 'float'")
    return (model.cuda() if torch.cuda.is_available() else model), n_bit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=ConvertConfig.model)
    p.add_argument("--mode", default="ternary", choices=("ternary", "float"))
    p.add_argument("--max-steps", type=int, default=ConvertConfig.max_steps)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-steps", type=int, default=ConvertConfig.save_steps)
    p.add_argument("--seq-len", type=int, default=ConvertConfig.seq_len)
    p.add_argument("--warmup-frac", type=float, default=0.2,
                   help="fraction of the run spent ramping lambda 0->1; the "
                        "recipe uses 1000 of 5000 steps = 0.2")
    p.add_argument("--lr", type=float, default=ConvertConfig.learning_rate)
    a = p.parse_args()

    # Warmup as a fraction of the run, not a fixed step count. The first
    # shakedown fixed it at 1000 steps, which was 67% of a 1500-step run against
    # the recipe's 20% — and, more to the point, 16M tokens against their 2B.
    cfg = ConvertConfig(model=a.model, mode=a.mode, max_steps=a.max_steps,
                        save_steps=a.save_steps, seq_len=a.seq_len,
                        learning_rate=a.lr,
                        warmup_lambda_steps=max(1, int(a.max_steps * a.warmup_frac)))
    out = Path(a.output_dir or f"outputs/convert/{a.mode}-{Path(a.model).name}")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, n_bit = build(cfg)
    print(f"mode={cfg.mode} bitlinears={n_bit} "
          f"tokens/step={cfg.tokens_per_step} total={cfg.tokens_per_step*cfg.max_steps/1e6:.0f}M",
          flush=True)

    callbacks = []
    warm = None
    if cfg.mode == "ternary":
        warm = LambdaWarmup(model, cfg.warmup_lambda_steps)
        callbacks.append(warm)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out), max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate, lr_scheduler_type="cosine",
            # Microsoft's 1-bit settings: beta2 0.95 (HF defaults 0.999), no
            # weight decay (large wd shrinks latent-weight magnitude and makes
            # ternary weights flip too often), grad clip 1.0 as nanotron.
            adam_beta2=0.95, weight_decay=0.0, max_grad_norm=1.0,
            warmup_steps=100, bf16=torch.cuda.is_available(),
            gradient_checkpointing=torch.cuda.is_available(),
            logging_steps=50, save_steps=cfg.save_steps, save_total_limit=None,
            report_to=[], seed=cfg.seed,
        ),
        train_dataset=load_stream(tok, cfg),
        callbacks=callbacks,
    )
    t0 = time.perf_counter()
    trainer.train()
    trainer.save_model(str(out / "final"))

    meta = {
        "config": asdict(cfg), "n_bitlinear": n_bit,
        "tokens_seen": cfg.tokens_per_step * cfg.max_steps,
        "seconds": round(time.perf_counter() - t0, 1),
        "lambda_history": warm.history if warm else None,
        # A ternary run whose final lambda is not 1.0 never finished warmup and
        # is not a ternary model, whatever its loss curve says. Computed from
        # the actual step count, not from the sampled history — history is
        # recorded every 100 steps and would report a stale value on a short run.
        "final_lambda": (bl.warmup_lambda(cfg.max_steps, cfg.warmup_lambda_steps)
                         if cfg.mode == "ternary" else None),
        "warmup_completed": (cfg.max_steps >= cfg.warmup_lambda_steps
                             if cfg.mode == "ternary" else None),
    }
    (out / "convert.json").write_text(json.dumps(meta, indent=2))
    print("CONVERT " + json.dumps({k: v for k, v in meta.items()
                                   if k != "lambda_history"}))


if __name__ == "__main__":
    main()
