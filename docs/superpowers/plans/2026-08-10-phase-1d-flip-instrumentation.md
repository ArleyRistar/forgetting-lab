# Phase 1d — Weight-state flip instrumentation (DESIGN CARD, awaiting approval)

**Status:** DRAFT v2 — not approved, no compute spent. Spec §3 gate.
**v2 (2026-08-10)** after review. Changes: the decomposition is now a real
partition (was a residual that could go negative and a test that could not fail);
the per-step burst moved out of LR warmup; the null arm's launch path is
specified and tested; distance predictors added to the *ternary* arm; disk
recounted 90 → 73 GB. Details at the end.

## The card, in the required form

**Hypothesis (what 1d tests — not H1 itself).** That weight-state flips can be
measured in a way that is *not* an artefact of the moving absmean scale. H1 says
flip fraction predicts forgetting; but the scale is per-tensor and recomputed
every forward pass, so any fine-tuning moves it and reclassifies every
near-threshold weight at once. **If flips cannot be separated into
threshold-driven and weight-driven components, H1 can confirm itself on a model
that learned nothing.** 1d builds the instrument and establishes its floor.
Phase 2 tests H1 with it.

**Method.** Offline analysis over latent checkpoints we already own (both 360M
twins), plus one short *null arm* per twin continuing on FineWeb-edu with **no
distribution shift**, densely checkpointed, to establish the flip rate
attributable to continued training alone.

**Metrics.** Per layer and per interval: flip fraction partitioned four ways,
flip persistence, distance-to-threshold histogram, zero occupancy; **latent
L2/cosine on both arms** plus KL-to-base as the competing predictors H1 must beat.

**Seeds.** Instrumentation is deterministic. The null arm uses **one** slice —
it establishes a floor, not an effect size — with a pre-registered escape hatch
(below). Spec §7's ≥3 seeds binds phase 2, not this.

**Estimated GPU-hours: ~4.5 thermal-derated** (cap ~40). Disk ~73 GB of 562 free.

---

## Why the definitions need this much care

Spec §6 1d asks for flip fraction, persistence and distance-to-threshold
histograms. Two things learned since change how it must be built:

1. **"Effective-value change" is the wrong operationalisation.** Effective values
   are `{-m, 0, +m}` with `m = mean|W|`, and `m` moves every step, so *every*
   weight's effective value changes constantly. The measurable quantity is the
   effective **state** `σ ∈ {-1, 0, +1}` — item 18's rename, in the code and not
   just the vocabulary.
2. **A raw flip count partly measures amount-of-training** (item 22), which
   correlates with forgetting trivially.

### Definitions

For tensor `W` at checkpoint `t`, with `s_t = 1 / mean|W_t|` (matching
`bitlinear.py:40` including its `clamp_(min=1e-5)`):

- **state** `σ(W, s) = clamp(round(W · s), -1, 1)`. Verified against
  `weight_quant`: effective value is `σ · m`, boundary at `|W·s| = 0.5`, and the
  `1.5` boundary is inert under the clamp. Implemented by **calling the same
  ops** as `weight_quant`, not by re-deriving with `>= 0.5` comparisons —
  `torch.round` is round-half-to-even and a reimplementation diverges at ties.
- **flip** at `(t-1, t)`: `σ(W_t, s_t) ≠ σ(W_{t-1}, s_{t-1})`.

### The partition (replaces v1's residual "interaction")

For every element that **actually flipped**, evaluate two counterfactuals —
weight-alone `σ(W_t, s_{t-1})` and scale-alone `σ(W_{t-1}, s_t)`, both against
`σ(W_{t-1}, s_{t-1})` — and assign it to exactly one class:

| class | weight-alone flips? | scale-alone flips? |
| --- | --- | --- |
| **weight-only** | yes | no |
| **scale-only** | no | yes |
| **redundant** | yes | yes |
| **joint** (needs both) | no | no |

These four are disjoint, non-negative, and sum **exactly** to the flip count — a
genuine partition, and every class is separately falsifiable. Reported alongside:
**cancelled** — elements where a counterfactual would have flipped but the actual
motion did not. v1 defined "interaction" as `total − weight − threshold`, which
can be negative and made its own consistency test unfalsifiable.

Both freeze conventions (`s_{t-1}` and `s_t`) are computed and reported; the
choice is asymmetric and the pair bounds its influence at zero cost.

- **persistence(k)**: of weights that flipped at `t`, the fraction still in the
  new state at `t+k`, for `k ∈ {1, 2, 4}`, per layer and pooled. (Settles spec
  §12 open question 4 — but see the resolution caveat in Task 2.)
- **distance-to-threshold**: histogram of `|W_t · s_t|` around 0.5. Scaled units
  are self-normalising (`mean|W·s| = 1` by construction), so the histogram is
  directly comparable across layers and checkpoints. Weights piling at 0.5 are
  Tequila's "deadzone boundary".
- **zero occupancy** per projection type. Baseline measured: **0.3254**.
- **competing predictors, both arms**: latent L2 and cosine of the parameter
  delta per layer, plus KL-to-base on a fixed probe set. H1 claims flips beat
  distance; measuring distance only on the float twin would confound predictor
  with arm, so the ternary arm gets it too. (This is item 19 in full; v1 did half.)

## Work

### Task 1 — `src/flab/flips.py`, tested on hand-built tensors

Numerics tested against hand-computed values, **not** through a model: the
phase-1b KL bug survived a model-based test because a small model's output
distribution hides almost everything.

Required tests, each targeting a distinct way this can be silently wrong:

- Scale changes, weights untouched → every flip is **scale-only**; weight-only
  and joint are exactly zero.
- Weights move, scale held fixed by construction → every flip **weight-only**.
- A weight both counterfactuals would flip → lands in **redundant**.
- A weight neither counterfactual flips but the combined motion does → **joint**.
  (The case v1's residual would have booked as negative interaction.)
- The four classes sum exactly to the flip count, on random tensors — non-vacuous
  now, because each class is computed independently rather than as a remainder.
- `σ(W, s)` agrees with `sign(weight_quant(W))` on random tensors **including
  exact ties at 0.5**, so the instrument matches the forward pass.
- A tensor against itself: flip fraction exactly 0.
- Zero occupancy on a hand-built tensor matches a hand count.

### Task 2 — run it over the conversion checkpoints (zero GPU)

Both twins, at steps **0 → 1000 → 2000 → 3000 → 4000**. Step 0 is the base
SmolLM2-360M weights, which *are* the initial latents (spec §6 1c), so it comes
free and adds an interval.

Two honesty constraints, both verified on disk:

- **`final/` is byte-identical to `checkpoint-4000`** (sha256 confirmed). There
  is no 4000→final interval; it would report zero flips and invite misreading.
- The 0→1000 interval spans the λ ramp (0→800 per `convert.json`), so it measures
  latent-state motion under partial quantisation, not fully-ternary motion. Label
  it as such.

Resolution limit stated plainly: these are 1000 steps apart, giving
per-1000-step fractions, **not** the per-step rate, and only `k ∈ {1, 2}`
persistence with a single sample at `k=2`. Spec §12 question 4 is therefore
settled mainly by the null arm, not by this task.

### Task 3 — the null arm (the only GPU cost; item 24)

Continue **each** twin on FineWeb-edu with no distribution shift, so phase 2 can
report task-induced flips over this floor. Analogous to the phase-1b null control
that showed the harness adds ~0 noise.

**Launch path, specified because the obvious one is silently wrong.** Running
this through `convert.py` would re-run λ warmup from 0 (`convert.py:71` →
`bitlinear.py:160`), so the arm would spend its first steps as a partially-float
model and every early "flip" would be conversion artefact. `load_stream` also has
**no skip support** (verified — it does not exist in any current code path). So:

- load via `flab.loading.load_converted` (λ=1, fp32), **constant λ=1**, with a
  per-step assert that λ is exactly 1 and `assert_ternary` at start and end;
- add stream-skip support, and verify the first block differs from training's;
- **fresh Adam and `warmup_steps=100`**: `final/` contains no `optimizer.pt`
  (verified), so optimizer state cannot be resumed. Stated here so the numbers
  are read as "continued training from cold optimizer", which is what they are.

Configuration: `.skip(300_000)` rows — past training's ~69,400 and the held-out
set's 250,000 window, disjoint from both by construction. 400 steps, conversion
hyperparameters (item 26's same-LR decision applies), **weights-only saves**.

- **Every 25 steps** → 16 checkpoints per arm.
- **Dense burst at steps 390–400**, checkpointed every step — *steady state*, not
  steps 1–10. Under `warmup_steps=100` the first ten steps run at 1e-6–1e-5, so a
  burst there would measure LR warmup and report a spuriously low per-step rate.
- Free analysis from the every-25 checkpoints: the **flip-fraction vs
  interval-length curve** (25/50/100/200-step lags). Flips do not add linearly
  when persistence < 1, so without this curve phase 2 cannot rescale a
  single-cadence floor to its own logging cadence.

Ternary ~400 × 7.5 s ≈ 0.85 h; float ~400 × 4.3 s ≈ 0.5 h.

**The floor is provisional.** Flip rate is strongly LR-dependent, and phase 2's
LR, schedule and cadence do not exist yet. This floor is conditional on *lr 1e-4,
fresh Adam, cosine over 400 steps, 25-step cadence*, and must be re-validated
(0.85 h) once the phase-2 card fixes hyperparameters. Its primary job here is
instrument validation: **is weight-driven signal measurable at all?**

**Escape hatch, pre-registered:** if phase-2 task-induced flips land within **3×**
this floor, a one-point floor cannot support the subtraction — re-run it with
≥2 further skip offsets (0.85 h each) before claiming an effect.

### Task 4 — the item-20 capability gate (eval only)

Item 31 scoped forgetting to what the **converted** model relearned, making this
a hard gate: a probe the ternary twin starts at chance on has nothing to forget,
and a flat curve would read as "no forgetting" when it means "no capability".

**Candidate probes** (already vendored, used in 1a/1b): `FOMC`, `ScienceQA`,
`Py150`, `NumGLUE-cm`, plus the `synth-*` pairs from `synthetic.py`.

**Operational criterion**, because "at chance" is undefined for NLL and CLAUDE.md
warns accuracies at this scale moved nothing beyond ~1.1 SE:

- primary, per probe: **answer NLL against a shuffled-answer control** on the
  same items. Keep the probe only if the true-answer NLL is better than the
  shuffled control by **≥3 SE** of the paired difference.
- secondary, reported not gating: chance-adjusted accuracy with its SE.
- n = 200 items per probe (the harness default), both twins.

Drop any probe failing the primary criterion, and record the drop. ~1.5 h.

### Task 5 — write up, including what the instrument cannot do

LAB-NOTES entry: conversion-run trajectory, the null floor and its conditionality,
the per-step rate, the interval-length curve, zero-occupancy motion, and the
surviving probe list.

## Budget

| item | GPU-h |
| --- | ---: |
| null arm, ternary (400 steps) | 0.85 |
| null arm, float (400 steps) | 0.50 |
| dense burst, both arms | 0.10 |
| item-20 capability gate | 1.50 |
| contingency (~50%, covers ~52 checkpoint writes) | 1.50 |
| **total** | **~4.5** |

Step times are measured soaked values, so the 1.9× derate is already inside them.
Tasks 1, 2 and 5 are CPU-only. VRAM risk ~nil: the null arm is
configuration-identical to runs with measured peaks of 6357 and 7303 MiB.

**Disk: ~73 GB** — (16 + 10) × 2 arms = **52** checkpoints, weights-only at 1.4 GB.
A full Trainer checkpoint is 2.3 GB (1.4 GB weights + 971 MB optimizer, measured);
the flip analysis never reads optimizer state, so saving it would waste 42%.
v1 said "~40 checkpoints ≈ 90 GB" — miscounted, and then called disk the number
worth watching.

## Gates and stopping conditions

- **Stop if weight-only + joint flips are < 5% of total flips in every interval**
  of both the conversion checkpoints and the null arm. That would mean flips at
  this budget are essentially threshold motion, H1 is not measurable as stated,
  and *that* is the result to report rather than proceeding to phase 2. The
  threshold is pre-registered here so it cannot be adjudicated after seeing the
  data; flip counts are deterministic, so there is no sampling noise to appeal to.
- **Investigate before proceeding if** the dense-burst per-step rate is **>10×
  away** from DQT's ~0.05%. Attach the expected direction: DQT measured at step
  2000 of *from-scratch* training at a much higher LR, so our continued-QAT run at
  1e-4 should sit **below** theirs. A rate far above would suggest 8-bit Adam
  interacting with near-threshold latents (item 29's trigger).
- **Item 20 is blocking**: if no candidate probe passes Task 4, phase 2 cannot
  produce a forgetting signal, and the response is distillation (item 17) or more
  conversion tokens — not proceeding.

## What this phase deliberately does not do

No sequential fine-tuning, no task sequence, no forgetting measurement — phase 2,
own card. No H2/H3 work. Item 23's attribution experiment stays deferred; only its
covariate is logged.

## v1 → v2 changelog

1. Dense burst moved from steps 1–10 to 390–400 — `warmup_steps=100` meant the
   burst would have measured LR warmup, biasing the per-step rate low and firing
   the DQT trigger spuriously.
2. Null-arm launch path specified (constant λ=1 via `flab.loading`, per-step
   assert, stream skip added, fresh-Adam stated) and given its own tests. v1 left
   the only GPU-spending task untested while `convert.py` would have silently
   re-run λ warmup from 0.
3. Three-way decomposition → four-way disjoint partition; the "total = sum" test
   was vacuous by construction and the residual could be negative.
4. Added `σ == sign(weight_quant)` tie-behaviour guard.
5. Latent L2/cosine added to the **ternary** arm — H1 claims flips beat distance,
   so the competing predictor must exist inside the arm (item 19 in full).
6. Stopping condition 1 made numeric (<5%) instead of "indistinguishable from zero".
7. Task 4 given a probe list and a ≥3 SE shuffled-control criterion.
8. Disk recounted 90 → 73 GB, 52 checkpoints, weights-only.
9. Step 0 added to Task 2; `final/`≡`checkpoint-4000` recorded; persistence
   resolution limits stated.
10. Floor declared provisional and LR-conditional, with the 3× escape hatch and
    the flip-vs-interval-length curve so phase 2 can rescale it.
