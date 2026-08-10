# Phase 2 — Does flip fraction predict forgetting? (DESIGN CARD, awaiting approval)

**Status:** DRAFT — not approved, no compute spent. Spec §3 gate.

## The card, in the required form

**Hypothesis (H1, spec §1).** The fraction of flipped ternary weights predicts
forgetting better than parameter-distance predicts it in the data-matched float
twin. **Phase 1d already found the null this must beat:** under diffuse
in-distribution drift, `flips ≈ 0.0002 · L2` to within ±15% across both arms, all
cadences and a 20× range of L2. If that relation survives task shift, flips are a
rescaled L2, carry no extra information, and **H1 is refuted** — which is a
reportable result, not a failed phase.

**Method.** Sequential fine-tuning A → B on both twins, over two paired synthetic
fact sets with analytically known outcomes, at three B-budgets to generate a
*range* of forgetting magnitudes, × 3 seeds. Forgetting of A is measured against
each twin's own post-A checkpoint (item 31).

**Metrics.** Forgetting of A = held-out answer NLL rise on A after training B.
Predictors, all computed on identical layers over identical checkpoint pairs:
flip fraction (four-way partition), flip persistence, per-layer flip
concentration, L2, cosine, KL-to-base (item 19 in full), and the float-sliver
embedding covariate (item 23).

**Seeds.** 3 on every result-bearing run (spec §7). Paired comparisons only.

**Estimated GPU-hours: ~6** (cap ~40). Disk ~140 GB of 491 free.

---

## Why the task sequence is synthetic, not TRACE

Not a shortcut — a consequence of two measured results:

* The item-20 gate (phase 1d) found the ternary twin cannot do **any** pretrained
  task: best 0.8 SE, Py150 answer NLL 8.29 against float's 2.45. A probe it
  starts at chance on has nothing to forget.
* Phase 1e found it learns fifty arbitrary associations to perfect held-out
  accuracy (NLL 5.7e-05, derangement +11.70 at 17.7 SE). Recall gone, plasticity
  intact.

Both arms must run the *same* sequence for the pair to mean anything, so the
sequence has to be tasks both twins can learn. The `synth-*` pairs — built in
phase 1b as controls — become the primary instrument, and they are a better fit
than that framing suggests:

* **`synth-conflict-a → -b` share a key namespace**, so learning B *must* destroy
  A. One distribution cannot concentrate on two values of the same key: if the
  model puts mass `p` on `v₂`, A's NLL is bounded below by `−log(1−p)`. **Large
  forgetting is the known answer**, analytically, before the run starts.
* **`synth-disjoint-a → -b` share nothing**, so nothing forces interference.
  **Zero forgetting is the known answer**, and whatever we measure instead is the
  harness's noise floor — the level below which phase 2 must not report effects.

Known answers at both ends is what makes this an instrument rather than a fishing
expedition.

## Why three B-budgets — H1 needs a range, not two points

H1 is a claim about *prediction*. Conflict (large) and disjoint (~0) give two
forgetting magnitudes, which cannot distinguish a good predictor from a bad one.
Varying B's training budget (**25 / 100 / 300 steps**) sweeps forgetting across
its range, turning H1 into a regression with enough points to compare predictors
honestly.

Branching keeps this cheap: train A **once** per (arm, pair, seed), checkpoint,
then run all three B-budgets from that checkpoint.

## Design

| factor | levels |
| --- | --- |
| arm | ternary twin, float twin |
| pair | conflict (forgetting forced), disjoint (noise floor) |
| B budget | 25, 100, 300 steps |
| seed | 0, 1, 2 |

12 A-runs (300 steps each) and 36 branched B-runs. Every run: lr 1e-4 **same in
both arms** (item 26), `completion_only` (item 13), full FT (an adapter would be
float and bypass the mechanism), 8-bit Adam, `max_length=256`, through
`sequential.py` — which phase 1e validated end to end on a ternary model.

**Checkpointing for flips:** save at A-end (= B-start) and B-end always, plus
every 25 steps within B. Flip analysis is offline; checkpoints are deleted once
their metrics are computed, so peak disk stays near 140 GB.

## The measurements

**Forgetting of A** — held-out answer NLL on A's eval split, after B, minus the
same at A-end. Reported in nats against the analytic chance level `log(8) =
2.0794`, so the scale is interpretable: ~0 is perfect retention, 2.08 means the
association is gone, ≫2.08 means the conflicting value was learned instead.

**Predictors, all on identical layers and checkpoint pairs (A-end → B-end):**

1. flip fraction, partitioned weight-only / scale-only / redundant / joint;
2. flip persistence at the run's own logging cadence (phase 1d: cadence must be
   stated or the number is meaningless);
3. **per-layer flip concentration** — the Gini or top-decile share of flips
   across the 224 tensors. This is the term most likely to decouple flips from
   L2, because L2 is a global magnitude and concentration is not;
4. L2 and cosine, per layer and global;
5. **KL-to-base on a fixed probe set** — owed since item 19 and not delivered in
   1d; the competing predictor H1 must beat besides L2;
6. embedding/head L2 as a covariate (item 23), since the tied embedding is 13.1%
   of the model, is fully plastic, and *is* the output head.

**The decisive analysis:** regress forgetting on each predictor across all 36
B-runs, within arm. H1 survives only if flip-based predictors beat L2 **in the
ternary arm** by more than L2 beats itself across arms. Report `flips / L2` for
every run: if it stays at 0.0002 ± 15%, say so plainly and retire H1 as stated.

**H2 comes free from the same runs** — magnitude and burstiness of the forgetting
curve at matched capability, read off the per-25-step checkpoints. No extra
compute. H3 stays out of scope (spec §7 escalation, own card).

## Gates and stopping conditions, pre-registered

* **Disjoint must show near-zero forgetting.** It is the noise floor and its
  answer is known. If disjoint forgetting is comparable to conflict forgetting,
  the harness is manufacturing an effect and no conflict result is
  interpretable. **Stop and debug**; report nothing.
* **Conflict must show large forgetting in *both* arms.** If the float twin does
  not forget A after learning conflicting B, the setup is broken, not
  interesting — that outcome is analytically excluded.
* **Re-validate the null floor at these hyperparameters first.** Phase 1d's
  0.00879%/step floor is explicitly conditional on lr 1e-4, fresh Adam, cosine
  over 400 steps, 25-step cadence. A ~0.5 GPU-h rerun at phase-2 settings comes
  before any B-run, and phase-2 flips are reported *over* it.
* **The 3× escape hatch** (phase 1d): if task-induced flips land within 3× the
  floor, a one-point floor cannot support the subtraction — re-run it at two more
  skip offsets before claiming an effect.
* **If `flips ≈ 0.0002 · L2` holds across every run**, H1 is refuted as stated.
  Report it, keep the instrument result, and do not go looking for a variant that
  rescues it after seeing the data.

## Budget

| item | GPU-h |
| --- | ---: |
| null-floor re-validation, both arms | 0.5 |
| 12 A-runs (300 steps: ternary ~10 min, float ~4 min) | 1.4 |
| 36 branched B-runs (425 steps per arm-pair-seed) | 2.0 |
| probes at every stage boundary (n=50, seconds each) | 0.1 |
| contingency (~50%) | 2.0 |
| **total** | **~6.0** |

Step times from phase 1e at `max_length=256`: 0.75 s/step float, ~2.1 s/step
ternary — the project's only short-sequence measurements, hence the 50%
contingency. Flip analysis, regressions and plots are CPU-only.

## What this card deliberately does not do

No TRACE tasks (the ternary twin cannot do them — item 20). No distillation
(item 17 is off the critical path since phase 1e). No H3, no from-scratch arm, no
rented-GPU arm — spec §7's escalation ladder, each needing its own card. No
domain-shift LM corpora: spec §7 lists them, but the gate result means the
ternary twin has no measurable capability on natural-language tasks to lose, so
they would measure conversion damage rather than forgetting.
