"""B-only control for the H2 falsifier: train B from the twin, never teaching A.

If v1 is still elevated above never-taught letters WITHOUT the model ever having
learned A, the elevation is a baseline property (letter priors, tokenizer
geometry) and not residual memory. This is the control that decides it.
"""
import argparse, torch
from transformers import AutoTokenizer
from flab import loading, sequential
from flab.runconfig import ProbeConfig, RunConfig, StageConfig, TrainSpec

p = argparse.ArgumentParser()
p.add_argument("--arm", choices=("ternary", "float"), required=True)
p.add_argument("--steps", type=int, default=300)
a = p.parse_args()
batch, accum = (4, 4) if a.arm == "ternary" else (2, 8)
src = f"outputs/convert/{a.arm}-360m/final"
out = f"outputs/phase2/{a.arm}-BONLY-s0-300"
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model, n = loading.load_converted(src, dtype=torch.float32)
if a.arm == "ternary":
    assert n > 0; loading.assert_ternary(model)
cfg = RunConfig(
    run_name=f"{a.arm}-bonly", model=src, mode="full", optim="adamw_bnb_8bit", seed=0,
    stages=(StageConfig(task="synth-conflict-b", learning_rate=1e-4, max_steps=a.steps),),
    train=TrainSpec(batch_size=batch, grad_accum=accum, max_length=256, completion_only=True),
    probe=ProbeConfig(tasks=["synth-conflict-a", "synth-conflict-b"], n_eval=50,
                      max_length=256, batch_size=4, reference_n=0))
sequential.run(cfg, out, model=model, tokenizer=tok)
if a.arm == "ternary": loading.assert_ternary(model)
print("BONLY DONE", out)
