"""Measure full-fine-tune VRAM for SmolLM2-360M (LAB-NOTES open item 4).

The §4 envelope's bytes-per-parameter table is computed from a LoRA run, which
only exercises optimizer state for 2.4% of the model. This runs a short *full*
fine-tune and reports where the memory actually goes, so the table can be
checked against reality. Batch/seq match the task-4 smoke run so the activation
term is comparable.

Usage: scripts/mem_probe.py --dtype bfloat16 --optim adamw_torch
"""
import argparse
import json

import torch
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from flab.data import load_smoltalk

MODEL = "HuggingFaceTB/SmolLM2-360M"


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


class MemProbe(TrainerCallback):
    """Snapshot the components once the optimizer has allocated its state."""

    def __init__(self) -> None:
        self.grads = 0
        self.rows: dict[str, int] = {}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kw):
        # Trainer zeroes grads with set_to_none after the step, so grads must be
        # read here rather than in on_step_end.
        self.grads = max(self.grads, tensor_bytes(p.grad for p in model.parameters()))

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kw):
        if state.global_step != 5 or self.rows:
            return
        opt = 0
        for st in optimizer.state.values():
            opt += tensor_bytes(v for v in st.values() if torch.is_tensor(v))
        self.rows = {
            "params": tensor_bytes(model.parameters()),
            "grads": self.grads,
            "optimizer": opt,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--optim", default="adamw_torch")
    p.add_argument("--max-steps", type=int, default=20)
    args = p.parse_args()

    data = load_smoltalk()
    probe = MemProbe()
    trainer = SFTTrainer(
        model=MODEL,
        args=SFTConfig(
            output_dir=f"outputs/memprobe/{args.dtype}-{args.optim}",
            max_steps=args.max_steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            # Always autocast: --dtype bfloat16 gives pure bf16 training, while
            # --dtype float32 gives mixed precision with fp32 master weights.
            # Those are the two configurations a QAT recipe would actually pick;
            # pure-fp32 compute is not one of them.
            bf16=True,
            gradient_checkpointing=True,
            max_length=1024,
            optim=args.optim,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            seed=0,
            model_init_kwargs={"dtype": args.dtype},
        ),
        train_dataset=data["train"],
        callbacks=[probe],
        # no peft_config: this is deliberately a full fine-tune
    )
    trainer.train()

    n = sum(q.numel() for q in trainer.model.parameters())
    peak = torch.cuda.max_memory_allocated()
    known = sum(probe.rows.values())
    print("PROBE " + json.dumps({
        "dtype": args.dtype,
        "optim": args.optim,
        "n_params": n,
        "warning": None if probe.grads else "grad hook never fired - grads undercounted",
        **probe.rows,
        "peak_allocated": peak,
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "activations_residual": peak - known,
        "bytes_per_param": round(known / n, 2),
    }))


if __name__ == "__main__":
    main()
