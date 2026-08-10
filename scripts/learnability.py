"""Phase 1e — can the converted twin still LEARN? (approved card 2026-08-10)

The item-20 gate asked *can it recall?* and got a clean no. This asks *can it
learn?*, which is the question item 31's scoping actually makes governing: if the
ternary twin can acquire a new fact set, phase 2 runs as teach-A → teach-B →
measure-forgetting-of-A with no new conversion.

Deliberately routed through `flab.sequential`, the phase-2 harness, which has
never trained a ternary model end to end. Validating it on a ~1 GPU-h job beats
discovering its faults inside a seven-run experiment.

Two metrics, because either alone can mislead:

* **held-out answer NLL vs the analytic chance level** `log(8) = 2.0794`.
  `synthetic.py` gives single-letter answers precisely so this number exists.
* **paired derangement delta**, as in the item-20 gate. A model can beat chance
  by learning the answer *marginal* ("answers are single letters A-H") while
  learning no mapping at all; the derangement holds the answer distribution
  fixed and breaks only the pairing, so it isolates the association.

Usage: uv run scripts/learnability.py --arm ternary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flab import loading, sequential, synthetic
from flab.runconfig import ProbeConfig, RunConfig, StageConfig, TrainSpec

TASK = "synth-conflict-a"
N_EVAL = 50            # all the held-out split holds
LEARNED_NLL = 1.4      # pre-registered on the card: a third under chance
LEARNED_SE = 3.0


def build_cfg(arm: str, steps: int, lr: float) -> RunConfig:
    return RunConfig(
        run_name=f"1e-{arm}",
        model=f"outputs/convert/{arm}-360m/final",
        mode="full",                       # no LoRA: an adapter would be float
        optim="adamw_bnb_8bit",            # not optional at 360M on 8 GB
        seed=0,
        stages=(StageConfig(task=TASK, learning_rate=lr, max_steps=steps),),
        train=TrainSpec(batch_size=4, grad_accum=4, max_length=256,
                        completion_only=True),   # item 13
        probe=ProbeConfig(tasks=[TASK], n_eval=N_EVAL, max_length=256,
                          batch_size=4, reference_n=0),
    )


def verify_completion_only(tokenizer) -> None:
    """Check the flag actually masks the prompt rather than trusting it.

    A silently-unmasked run would train on the prompt distribution too, which
    is a different experiment and would not announce itself.
    """
    from flab import probes, trace

    ex, _ = trace.load_probe_examples(TASK, n_eval=1, seed=0)
    ids, labels, _ = probes._encode(tokenizer, ex[0]["prompt"], ex[0]["answer"], 256)
    supervised = [i for i, l in enumerate(labels) if l != -100]
    if not supervised:
        raise RuntimeError("no supervised tokens at all")
    if supervised[0] < len(ids) - 8:
        raise RuntimeError(
            f"prompt does not look masked: {len(supervised)} of {len(ids)} tokens "
            "carry labels; expected only the answer tail")
    print(f"  completion_only verified: {len(supervised)} supervised token(s) "
          f"of {len(ids)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("ternary", "float"), required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--run-dir", default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer

    cfg = build_cfg(a.arm, a.steps, a.lr)
    run_dir = Path(a.run_dir or f"outputs/learnability/{a.arm}")
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    verify_completion_only(tok)

    model, n_bit = loading.load_converted(cfg.model, dtype=torch.float32)
    if a.arm == "ternary":
        if n_bit == 0:
            raise SystemExit("ternary twin loaded with 0 BitLinear layers")
        loading.assert_ternary(model)
        print(f"  {n_bit} BitLinear layers verified three-valued at λ=1", flush=True)

    state = sequential.run(cfg, run_dir, model=model, tokenizer=tok)

    if a.arm == "ternary":
        # Still ternary AFTER training, not merely at the start. A run that
        # silently de-ternarised would otherwise be written up as ternary.
        loading.assert_ternary(model)
        print("  still ternary after training", flush=True)

    out = {"arm": a.arm, "task": TASK, "steps": a.steps, "lr": a.lr,
           "chance_nll": synthetic.chance_nll(),
           "learned_threshold_nll": LEARNED_NLL,
           "baseline_probe": state.baseline_probe,
           "stages": [s.probe for s in state.stages]}
    Path(run_dir / "learnability.json").write_text(json.dumps(out, indent=2, default=str))
    print("LEARNABILITY " + json.dumps({"arm": a.arm, "chance": out["chance_nll"]}))


if __name__ == "__main__":
    main()
