# Phase 0 — Smoke-Test Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the whole training stack on the lab box with one reproducible LoRA fine-tune of SmolLM2-360M, evaluated before/after — the spec's §5 "hello training" deliverable.

**Architecture:** A tiny `flab` Python package (uv-managed, src layout): a data module that formats a smol-smoltalk subset to plain text, a TRL `SFTTrainer` entrypoint with LoRA + resume support, an lm-evaluation-harness wrapper script, and a GPU burn script that doubles as the bring-up thermal verification.

**Tech Stack:** Python 3.12+, uv, PyTorch (CUDA wheel from PyPI), transformers, TRL ≥0.20, peft, datasets, lm-eval, tensorboard, pytest.

**Precondition:** `docs/bringup-checklist.md` is fully verified (GPU visible, headless, repo cloned at `~/forgetting-lab` on `gs66-lab`). This plan executes ON THE LAB BOX.

## Global Constraints

- bf16 + gradient checkpointing on all training (spec §4).
- Development runs use **seed 0**, single model SmolLM2-360M (spec §6 1a).
- Smoke-test compute must stay well under the 40 GPU-h design-card cap (spec §3); budget here ≈ 2 GPU-h total.
- `outputs/` is never committed; code and `uv.lock` are.
- Long-running commands (training, eval, burn) run inside `tmux`.
- Python only via `uv run` — no system pip, no conda.
- Commits go straight to `main` (personal-repo convention).

---

### Task 1: Project scaffold + environment

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/flab/__init__.py` (empty)

**Interfaces:**
- Produces: importable package `flab`; `uv run` env with CUDA-enabled torch.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "flab"
version = "0.1.0"
description = "forgetting-lab: catastrophic forgetting in ternary LLMs"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.6",
    "transformers>=4.51",
    "datasets>=3.2",
    "peft>=0.14",
    "trl>=0.20",
    "accelerate>=1.2",
    "lm-eval>=0.4.8",
    "tensorboard>=2.18",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/flab"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
outputs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create the package and sync**

Run: `mkdir -p src/flab scripts tests && touch src/flab/__init__.py && uv sync`
Expected: resolves and installs without error (first sync downloads ~3 GB of wheels).

- [ ] **Step 4: Verify CUDA**

Run: `uv run python -c "import torch; print(torch.cuda.get_device_name(0))"`
Expected: prints a name containing `3070 Ti`. If `cuda.is_available()` is False, stop and debug the driver before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/
git commit -m "feat: scaffold flab package with uv environment"
```

### Task 2: GPU burn — thermal + power-cap verification

**Files:**
- Create: `scripts/burn.py`
- Create: `docs/LAB-NOTES.md`

**Interfaces:**
- Produces: `docs/LAB-NOTES.md` (the running lab notebook all later tasks append to).

- [ ] **Step 1: Write `scripts/burn.py`**

```python
"""10-minute GPU burn: large matmuls while logging temp/power/clocks.

Verifies the bring-up thermal setup (power cap, elevation) under sustained
load. Usage: uv run scripts/burn.py [minutes]
"""
import subprocess
import sys
import time

import torch


def gpu_stats() -> str:
    q = "temperature.gpu,power.draw,clocks.sm,utilization.gpu"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def main(minutes: float) -> None:
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    end = time.time() + minutes * 60
    i = 0
    while time.time() < end:
        a = (a @ b).tanh()  # keep values bounded so the loop can run forever
        i += 1
        if i % 200 == 0:
            torch.cuda.synchronize()
            print(f"[{time.strftime('%H:%M:%S')}] iter={i} {gpu_stats()}", flush=True)
    torch.cuda.synchronize()
    print("done:", gpu_stats())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 10)
```

- [ ] **Step 2: Run the burn in tmux**

Run: `tmux new -s burn 'uv run scripts/burn.py 10 | tee /tmp/burn.log'`
Expected: 10 minutes of stat lines; temperature climbs then **plateaus below ~87 °C** with no crash. If it passes 90 °C or the machine dies, stop — fix cooling/power-cap before any training.

- [ ] **Step 3: Record the result in the lab notebook**

Create `docs/LAB-NOTES.md`:

```markdown
# forgetting-lab — lab notes

## 2026-MM-DD — bring-up burn test (phase 0, task 2)

10-min bf16 matmul burn (`scripts/burn.py`):
- power limit in effect: <value from nvidia-smi> W
- temperature plateau: <value> °C
- sustained SM clock: <value> MHz

Verdict: <ok to train unattended / needs attention because …>
```

Fill in the real numbers from `/tmp/burn.log`.

- [ ] **Step 4: Commit**

```bash
git add scripts/burn.py docs/LAB-NOTES.md
git commit -m "feat: add GPU burn script; record bring-up thermal results"
```

### Task 3: Data module (TDD)

**Files:**
- Create: `tests/test_data.py`
- Create: `src/flab/data.py`

**Interfaces:**
- Produces: `format_messages(messages: list[dict]) -> str` and
  `load_smoltalk(n_train: int = 4000, n_eval: int = 200, seed: int = 0) -> DatasetDict`
  with splits `train`/`eval`, each having a single `text` column. Task 4 consumes `load_smoltalk`.

- [ ] **Step 1: Write the failing test**

```python
from flab.data import format_messages


def test_format_messages_renders_roles_in_order():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = format_messages(msgs)
    assert out == "<|user|>\nhi\n<|assistant|>\nhello\n<|end|>"


def test_format_messages_supports_system_role():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    out = format_messages(msgs)
    assert out.startswith("<|system|>\nbe brief\n<|user|>")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flab.data'`

- [ ] **Step 3: Write `src/flab/data.py`**

```python
"""Load and format the smoke-test SFT dataset (smol-smoltalk subset)."""
from datasets import DatasetDict, load_dataset

TAGS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def format_messages(messages: list[dict]) -> str:
    parts = [f"{TAGS[m['role']]}\n{m['content']}" for m in messages]
    return "\n".join(parts) + "\n<|end|>"


def load_smoltalk(n_train: int = 4000, n_eval: int = 200, seed: int = 0) -> DatasetDict:
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_train + n_eval))
    ds = ds.map(
        lambda ex: {"text": format_messages(ex["messages"])},
        remove_columns=ds.column_names,
    )
    return DatasetDict(
        train=ds.select(range(n_train)),
        eval=ds.select(range(n_train, n_train + n_eval)),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 2 passed.

- [ ] **Step 5: Sanity-check the real dataset loads (downloads ~a few hundred MB)**

Run: `uv run python -c "from flab.data import load_smoltalk; d = load_smoltalk(); print(d); print(d['train'][0]['text'][:200])"`
Expected: DatasetDict with train=4000/eval=200 rows, sample text starts with a `<|...|>` tag. If the dataset id 404s, check its HF page for a rename before improvising a substitute — record whatever is used in LAB-NOTES.

- [ ] **Step 6: Commit**

```bash
git add tests/test_data.py src/flab/data.py
git commit -m "feat: smol-smoltalk data module with plain-text chat formatting"
```

### Task 4: Training entrypoint + the smoke run

**Files:**
- Create: `src/flab/train.py`

**Interfaces:**
- Consumes: `load_smoltalk()` from Task 3.
- Produces: CLI `uv run python -m flab.train [--output-dir D] [--max-steps N] [--resume]`; LoRA adapter saved to `<output-dir>/final`. Task 5 consumes that adapter path.

- [ ] **Step 1: Write `src/flab/train.py`**

```python
"""LoRA SFT smoke test: SmolLM2-360M on a smol-smoltalk subset (spec §5)."""
import argparse

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from flab.data import load_smoltalk

MODEL = "HuggingFaceTB/SmolLM2-360M"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/smoke")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    data = load_smoltalk()
    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        max_length=1024,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        report_to="tensorboard",
        seed=0,
        model_init_kwargs={"torch_dtype": "bfloat16"},
    )
    trainer = SFTTrainer(
        model=MODEL,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
```

Note: field names (`max_length`, `eval_strategy`, `model_init_kwargs`) are for TRL ≥0.20 as pinned in Task 1. If SFTConfig rejects one, check `uv run python -c "import trl; print(trl.__version__)"` and the installed version's SFTConfig signature — adjust the field name, not the pin.

- [ ] **Step 2: 20-step dry run**

Run: `uv run python -m flab.train --output-dir outputs/dry --max-steps 20`
Expected: completes in a few minutes; log shows loss at step 10 and 20; no OOM. Typical starting loss ≈ 2–3, and it should already tick down.

- [ ] **Step 3: Full smoke run in tmux**

Run: `tmux new -s smoke 'uv run python -m flab.train 2>&1 | tee /tmp/smoke.log'`
Expected: ~400 steps in well under 2 h. Watch the first minutes with `nvidia-smi` — VRAM should sit comfortably under 8 GB (if OOM, halve `per_device_train_batch_size` and double `gradient_accumulation_steps`).

- [ ] **Step 4: Verify the loss curve in TensorBoard**

Run: `uv run tensorboard --logdir outputs/smoke --bind_all` then browse `http://gs66-lab.local:6006` from the Zenbook.
Expected: train loss decreasing, eval loss decreasing; no divergence.

- [ ] **Step 5: Verify crash-resume**

Run: `uv run python -m flab.train --output-dir outputs/resume-test --max-steps 250`, kill it with Ctrl-C somewhere after step 100 (past a save at step 100), then rerun with `--resume`.
Expected: training restarts from the saved checkpoint's step (log says "Continuing training from checkpoint"), not from 0, and finishes at 250.

- [ ] **Step 6: Commit**

```bash
git add src/flab/train.py
git commit -m "feat: LoRA SFT smoke-test entrypoint with resume support"
```

### Task 5: Before/after evaluation

**Files:**
- Create: `scripts/eval.sh`
- Modify: `docs/LAB-NOTES.md` (append results)

**Interfaces:**
- Consumes: adapter at `outputs/smoke/final` from Task 4.
- Produces: eval JSON under `outputs/eval/<tag>/`; results table in LAB-NOTES.

- [ ] **Step 1: Write `scripts/eval.sh`**

```bash
#!/usr/bin/env bash
# Evaluate SmolLM2-360M with lm-evaluation-harness.
# Usage: scripts/eval.sh <run-tag> [peft-adapter-path]
set -euo pipefail
TAG=$1
PEFT=${2:+,peft=$2}
uv run lm_eval --model hf \
  --model_args "pretrained=HuggingFaceTB/SmolLM2-360M,dtype=bfloat16${PEFT}" \
  --tasks arc_easy,hellaswag,ifeval \
  --batch_size auto \
  --output_path "outputs/eval/${TAG}"
```

Run: `chmod +x scripts/eval.sh`

- [ ] **Step 2: Evaluate the base model**

Run: `tmux new -s eval 'scripts/eval.sh base 2>&1 | tee /tmp/eval-base.log'`
Expected: finishes in well under an hour on GPU; prints a results table. arc_easy ~0.5–0.6 acc, hellaswag ~0.4–0.55 acc_norm, ifeval near zero for the base model — exact values don't matter, only that they're recorded.

- [ ] **Step 3: Evaluate the fine-tuned model**

Run: `tmux new -s eval 'scripts/eval.sh smoke outputs/smoke/final 2>&1 | tee /tmp/eval-smoke.log'`
Expected: completes; ifeval should move visibly above the base score (the model just learned instruction-following format); arc_easy/hellaswag should move little.

- [ ] **Step 4: Append the results table to `docs/LAB-NOTES.md`**

```markdown
## 2026-MM-DD — smoke run before/after (phase 0, tasks 4–5)

400-step LoRA SFT of SmolLM2-360M on smol-smoltalk[:4000], seed 0.

| task | base | after SFT |
| --- | --- | --- |
| arc_easy (acc) | <val> | <val> |
| hellaswag (acc_norm) | <val> | <val> |
| ifeval (inst_level_loose_acc) | <val> | <val> |

Wall-clock: train <val> min, eval <val> min each. Peak VRAM: <val> GB.
Notes: <anything surprising>
```

Fill in real values from the two eval logs.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval.sh docs/LAB-NOTES.md
git commit -m "feat: lm-eval wrapper; record smoke-run before/after results"
```

### Task 6: Walkthrough + reproducibility gate

**Files:**
- Modify: `docs/LAB-NOTES.md` (walkthrough notes)

**Interfaces:**
- Consumes: everything above. This task closes phase 0 (spec §5 deliverable).

- [ ] **Step 1: Guided walkthrough (interactive — do this WITH Arley, not for him)**

Walk through `src/flab/train.py` and `src/flab/data.py` end to end in the session: what a LoRA adapter actually is (which matrices, why r=16), what one optimizer step does at bf16, why gradient checkpointing trades compute for memory, what the chat-format tags do, and what `save_steps`/resume actually persist. Stop at each block until Arley says it's clear. Append a short "questions that came up" list to `docs/LAB-NOTES.md`.

- [ ] **Step 2: Reproducibility check from a clean clone**

Run:
```bash
git clone ~/forgetting-lab /tmp/flab-repro && cd /tmp/flab-repro
uv sync && uv run pytest -q
uv run python -m flab.train --output-dir /tmp/flab-repro-out --max-steps 20
```
Expected: tests pass and the 20-step run completes from a pristine clone — the "reproducible hello-training run" deliverable. Delete `/tmp/flab-repro*` afterwards.

- [ ] **Step 3: Commit and mark phase 0 done**

```bash
git add docs/LAB-NOTES.md
git commit -m "docs: phase-0 walkthrough notes; smoke test reproducible from clean clone"
```

Phase 0 is complete. Next: phase-1 design cards (spec §6), starting with 1a (harness) — that gets its own plan.
