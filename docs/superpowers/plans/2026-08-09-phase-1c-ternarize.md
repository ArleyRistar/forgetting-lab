# Phase 1c — Ternarize-and-Own Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the matched pair the whole project rests on — SmolLM2-360M
converted to ternary via continued QAT, and a float twin continued-trained on
the *same tokens* — with full latent checkpoints saved throughout. Spec §6 1c.

**Why it matters:** no public ternary checkpoint releases usable QAT latent
weights (verified 2026-08-05), so converting one ourselves is the only way to
own the latent trajectory phase 1d measures flips on.

**Precondition:** 1a and 1b complete. Calibration gate passed 2026-08-09.

## The recipe (from the HF Llama3→1.58bit work, fetched 2026-08-09)

```python
# weights: absmean scale, ternarise
scale_w = 1.0 / w.abs().mean().clamp_(min=1e-5)
w_q = clamp(round(w * scale_w), -1, 1) / scale_w

# activations: per-token absmax to int8
scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
x_q = (x * scale_x).round().clamp_(-128, 127) / scale_x

# straight-through estimator, with warmup
x_quant = x + lambda_ * (quant(x) - x).detach()
lambda_ = min(step / 1000, 1)
```

- BitLinear replaces `nn.Linear` in **attention and FFN only** — not embeddings,
  not `lm_head`, not norms.
- **LayerNorm before activation quantisation is essential**, per the writeup.
- Theirs: lr 1e-4, batch 2M tokens, 5000 steps = **10B tokens**, FineWeb-edu.

## The binding constraint: we have ~25× less compute than the recipe

At effective batch 16 × seq 1024 = 16k tokens/step, and ~4.5 s/step measured:

| budget | steps | tokens |
| --- | --- | --- |
| 30 GPU-h | ~24,000 | **~390M** |
| 80 GPU-h | ~64,000 | **~1.0B** |

Against their 10B. **The conversion will be worse than theirs and that is
expected** — spec §6 1c says to *measure and report the conversion quality gap*,
not to minimise it. The gap is a documented property of our pair, and phase 2
measures forgetting relative to each twin's own post-conversion baseline (§9)
precisely so an absolute gap does not confound the comparison.

The 1000-step lambda warmup is ~4% of our shortest budget and ~1.6% of theirs.
Keep it at 1000 steps rather than scaling it down: it exists to stop the model
collapsing at the moment quantisation switches on, and that risk does not shrink
with a smaller budget.

---

### Task 1: BitLinear, tested on numerics before anything trains

**Files:** create `src/flab/bitlinear.py`, `tests/test_bitlinear.py`

- [ ] **Step 1: Implement the quantisers as standalone functions**

`weight_quant(w)` and `activation_quant(x)` exactly as above, separate from the
module so they can be tested against hand-computed values — the same discipline
that caught the KL direction error in 1b, where testing through a model hid the
bug.

- [ ] **Step 2: Test the numerics directly**

Assert on hand-built tensors: weights land in {-s, 0, +s}; a uniform weight
matrix ternarises to all ±1 scaled; the absmean scale matches `1/mean(|w|)`;
activations clamp at ±127/scale. Assert **the STE passes gradients through
unchanged** (`d out/d in == 1` where the quantiser is not saturated) — that is
the one property the whole recipe depends on and it is invisible in forward
outputs.

- [ ] **Step 3: `lambda_ = 0` must be a mathematical no-op**

At warmup start the layer must be **bit-identical to the original `nn.Linear`**.
This is the 1b `disable_adapter` check again: if it is not exactly equal, the
conversion starts from something other than the float model and "the float
weights *are* the initial latent weights" stops being true.

- [ ] **Step 4: `lambda_ = 1` must actually ternarise**

Effective weights take at most 3 distinct values per tensor. Guards against a
no-op that silently trains a float model and reports it as ternary.

---

### Task 2: Injection into SmolLM2

**Files:** modify `src/flab/bitlinear.py`; `tests/test_bitlinear.py`

- [ ] **Step 1: Replace attention and FFN linears, leave the rest**

`q/k/v/o_proj`, `gate/up/down_proj`. **Not** `embed_tokens`, **not** `lm_head`.
Assert the replaced count matches `num_layers × 7` and that embeddings and head
are untouched objects, not merely equal.

- [ ] **Step 2: Confirm the parameter count is unchanged**

Latent weights stay fp32/bf16 tensors of the same shape; ternarisation happens
in the forward pass. A changed parameter count means the model was rebuilt
rather than converted, and the latent trajectory would not be the float model's.

- [ ] **Step 3: Verify RMSNorm already precedes the quantised path**

The writeup calls normalisation before activation quantisation essential. SmolLM2
is a Llama-style pre-norm architecture, so this should hold already — **check it
rather than assume**, and record what was checked.

---

### Task 3: Conversion training

**Files:** create `src/flab/convert.py`, `configs/convert-135m.yaml`

- [ ] **Step 1: Lambda warmup as a callback**

`lambda_ = min(step/1000, 1)`, pushed into every BitLinear each step. Log it
alongside loss — a warmup that silently never advances would look like a
model that simply trains badly.

- [ ] **Step 2: Data — FineWeb-edu, streamed**

Their corpus. Streaming avoids materialising a large dataset on the box. Record
the exact number of tokens consumed: the float twin must see **the same tokens**
(spec §6 1c), so the token stream is part of the experiment, not a detail.

- [ ] **Step 3: Save full latent checkpoints**

Phase 1d measures flips on the latent trajectory, so checkpoints must contain
the **latent** weights, not the ternarised effective weights. Disk is the
constraint flagged in open item 3: ~1.4 GB per checkpoint at 360M in bf16.
Decide a retention policy before starting, not after filling the disk.

---

### Task 4: Shakedown at 135M

- [ ] **Step 1: Convert SmolLM2-135M, short run**

Spec §9 names QAT fiddliness as the main technical risk and budgets one restart.
The 135M shakedown is where that restart happens cheaply.

- [ ] **Step 2: Watch for the failure modes that matter**

Loss spiking at the moment lambda leaves 0; loss flat (warmup not applied);
effective weights not actually ternary at lambda=1. Each has a distinct
signature and each would otherwise be discovered at 360M scale.

- [ ] **Step 3: Measure the ternary QAT memory rows (closes open item 7)**

Everything measured so far is float training. `scripts/mem_probe.py` extended to
the BitLinear path gives the materialised-quantised-tensor term the §4 table
still computes rather than measures.

---

### Task 5: The 360M pair

- [ ] **Step 1: Ternary conversion run**
- [ ] **Step 2: Float twin on the same tokens**

Data-matched is the point (§9): if the twins see different tokens, the
conversion corpus confounds every later comparison.

- [ ] **Step 3: Report the conversion quality gap**

Held-out NLL and generative exact-match for both twins against the original
SmolLM2-360M. Both observables — 1b established they can disagree, and by how
much.

---

## Decisions needed

1. **Token budget** — 30 GPU-h (~390M tokens) or 80 (~1.0B)? Both are far below
   the recipe's 10B. Recommend starting at 30 for the 360M pair and extending if
   the conversion gap looks recoverable, since an under-trained pair is still a
   valid *matched* pair for measuring forgetting.
2. **Latent checkpoint retention** — open item 3, now blocking. At ~1.4 GB each,
   every-1000-steps over 24k steps is ~34 GB per twin. Feasible on the 1 TB NVMe
   but should be chosen.
3. **`completion_only`** (open item 13) does not apply here — conversion is
   plain LM training on FineWeb-edu, no prompt/answer split.
