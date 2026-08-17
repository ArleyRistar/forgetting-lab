# Transferable findings

Written for whoever — person or agent — lands here looking for something usable,
without reading a 3,000-line lab notebook. Dense on purpose. Every entry gives the
number, the conditions it holds under, and where to verify it.

**Status of this repo:** the experiments are finished and nothing further is
planned. A blog post was drafted (`docs/blog/01-ternary-conversion.md`) and not
published. The findings below stand on their own; take them.

**Scale caveat that applies to everything here:** SmolLM2-360M, 65.5M conversion
tokens, one 8 GB laptop GPU. That is ~150× less data than the reference recipe.
Read these as "what happens at a hobby budget", not as limits of the method.

---

## Practically useful, most first

### 1. bf16 latent weights silently kill QAT conversion at small batch sizes

At lr 1e-4, **85.4%** of Adam updates round to zero; at lr 2e-5, **96.3%**.
Conversion fails with a loss curve that merely looks like bad training. Lowering
the learning rate makes it *worse*, which is what makes it hard to diagnose.

Changing only the dtype to fp32 (learning rate held at 1e-4) fixed it.

- Conditional on **bf16 + small batch + lr 1e-4**. At Microsoft's 1M-token batch
  and lr 1.5e-3 the per-step update is far larger and this does not bite.
- Every reference implementation trains an fp32 master (nanotron defaults
  `accumulate_grad_in_fp32: true`) — the technique is standard, the *failure mode*
  is not documented anywhere we found.
- Verify: `docs/LAB-NOTES.md`, "the shakedowns failed because latent weights were
  bf16".

### 2. Ternary forward passes are not batch-invariant

Scoring identical prompts at batch 1 vs batch 8: **median 0.34 nats** drift in a
ternary model, **1.1e-5** in its float twin. ~30,000×.

- **Step change, not a gradient** — batch 2 is already as bad as batch 32. Smaller
  batches do not help; only batch 1 does.
- **Not padding** — equal-length inputs, no padding at all, give 0.342.
- **Activation quantisation is the cause.** Disable it → 1.5e-5 (float levels).
  Disable weight quantisation instead → 0.062 remains.
- **Not the known batch-coupling bug.** Our scale is per-token (`dim=-1`);
  measured, a row quantised alone is bit-identical to the same row batched with
  rows 120× larger. Distinct from cross-batch scales
  ([Quantamination](https://arxiv.org/abs/2604.26505)).
- Mechanism: cuBLAS gives ~1e-7 per-matmul differences by batch shape; `round()`
  in activation quantisation amplifies across int8 levels by ~1e5; 224 layers
  compound it. Amplification-as-principle is
  [Defensive Quantization](https://arxiv.org/abs/1904.08444) (ICLR 2019).
- **Greedy output is unaffected** — 0 argmax flips in 64, for both shipped models.
  (The weight-quant-off *diagnostic* ablation flipped 18/64.)
- **Scope: simulated quantisation in float kernels** (QAT, most research code).
  True integer pipelines may differ —
  [arXiv 2607.23227](https://arxiv.org/abs/2607.23227) argues INT8 on ARM is
  dispatch-invariant.
- **What to do:** score low-bit models at batch 1 for anything likelihood-based,
  or report the batch size as part of the measurement.
- Verify: `results/ternary-batch-stability.json`,
  `scripts/ternary_batch_stability.py`.

### 3. `lm_eval --batch_size auto` is allocator-state-dependent

Observed: 8 OOM backoffs from 4.83 GB → 2.42 GB, settling on batch **2**, while
the run header reported `batch_size: auto (64)`. Free-VRAM readings differed
between consecutive failures.

Combined with (2): **two `lm_eval` runs of identical ternary weights can be scored
at different batch sizes and disagree** by more than many published deltas. Pin
`--batch_size` to an integer for anything paired or repeated.

### 4. lm-eval's logged loglikelihoods are bf16-quantised

Under `dtype=bfloat16`, `resps` values are exact multiples of **0.125** at |LL|
16–32 and **0.25** at 32–64. A per-item logprob shift smaller than the local step
is not representable at all — a hard floor under any paired logprob-margin
instrument. Not yet characterised: whether a typical LoRA moves margins above it.

### 5. Perplexity recovery is not capability recovery

A converted 360M ternary model reached held-out loss 5.13 (ppl 169) — bad but
not absurd — and had **zero measurable capability**: 0.1–0.8 SE discrimination
against shuffled answers on every pretrained task, where the base model scored
6.1 and 9.4 SE and the float twin 5.5 and 9.2.

Py150 answer NLL: base 2.31, float twin 2.45, **ternary 8.29**; discrimination
+1.69 nats → **+0.017**.

If you convert a small model on a small budget, perplexity will not tell you this
happened. Verify: `results/null-capability-gate.json`.

### 6. Recall can be gone while plasticity is intact

The same checkpoint that discriminates on nothing memorises **50 arbitrary
key→value associations to perfect held-out accuracy** (NLL 5.7e-05, shuffled-
control margin +11.70 at 17.7 SE). Conversion at this budget destroyed what the
model knew, not its ability to learn.

Practical reading: if you are converting in order to fine-tune on your own task
anyway, the capability loss may matter less than the perplexity suggests.

---

## Method notes worth stealing

### The data-matched float twin

Train the *same* base model on the *same tokens in the same order* in float, as a
control. It is the only way to separate "what ternarisation cost" from "what the
extra training did". Ours landed **+0.0116 nats** from base, so the 2.59-nat gap
is attributable to ternarisation.

[Spectra](https://arxiv.org/abs/2407.12327) owns the matched-precision-suite idea;
what we could not find was a *conversion* study running the control at a matched
token budget. `src/flab/convert.py --mode {ternary,float}` runs both.

### A "disjoint" task pair is NOT a noise floor

Two synthetic tasks sharing no keys still produced **~2 nats** of real forgetting,
while the true harness floor — training on the *same* task again — was **0.0126**.
Using the disjoint pair as a baseline, as our own spec proposed, would have
subtracted a genuine effect out of every result. If you want a noise floor, use
same-task continuation.

### Capability gates need a positive control

"The model scores 0.8 SE" is ambiguous between *no capability* and *broken test*.
Running the untouched base model through the same gate disambiguates it, and cost
~0.5 GPU-h. It also revealed FOMC failing for *every* arm — a bad probe, not
evidence about any model.

### Guard likelihood-only capability claims in code

`src/flab/claims.py` refuses to report a forgetting claim when accuracy is at
floor in every arm, or when NLL sits >2 nats above a task's chance level. Written
after we retracted a claim that violated both. ~100 lines, tests included.

---

## Dead ends — do not rebuild these

### Weight-state flips are a fixed rescaling of L2 distance

`flips / L2` = **0.0002 ± 15%** across both arms, every cadence (1-step, 25-step,
1000-step), a 20× range of L2, and under task shift as well as diffuse drift. Flip
fraction cannot predict anything parameter distance does not.

Directly: in the ternary arm, flips and L2 correlate with forgetting at +0.5952
and +0.6006 — indistinguishable, flips marginally worse. **This was the project's
primary hypothesis and it is refuted.** The instrument still works
(`src/flab/flips.py`, four-way causal partition, planted-effect tests) if you want
it for something else.

### The threshold-motion confound is negligible in practice

The absmean scale is recomputed every forward pass, so a weight's state can flip
without the weight moving — which sounds like it should contaminate any flip
metric. Measured: scale-driven flips are **0% per step**, rising only to 0.0084%
of flips at 1000-step intervals. Worth having checked; not worth designing around.

### "Ternary models forget less" — RETRACTED

Published and withdrawn. The ~6-nat gap was measured where task-A accuracy was
**0.00 in both arms**: both models had forgotten completely, and the difference
lived in the log-probability of a token neither would ever emit. Where retention
was measurable, the direction reversed (float 0.60 vs ternary 0.40).

**The transferable lesson:** a likelihood difference between two models is not a
capability difference, and the further you are from chance the less it means.
Check the behavioural number before believing the probabilistic one.

### Cross-arm comparisons in log units are unsafe here

A follow-up found a sub-behavioural trace surviving overwriting, ~2× larger in the
float twin — until a scale-free rank statistic showed the ternary effect was not
established at all (t=1.8, CI spanning zero, against t=20 for float). The arms'
log scales differ by ~2.5 nats on never-taught letters. If you compare two models
in log-probability units, use a rank-based statistic to check.

---

## Where the numbers live

| what | where |
| --- | --- |
| every measurement, newest at the bottom, retractions included | `docs/LAB-NOTES.md` |
| the JSON behind every published number | `results/` |
| the unpublished write-up | `docs/blog/01-ternary-conversion.md` |
| design cards, one per experiment | `docs/superpowers/plans/` |
| the spec, incl. killed alternatives | `docs/superpowers/specs/` |
