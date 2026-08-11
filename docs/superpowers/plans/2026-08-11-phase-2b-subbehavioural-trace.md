# Phase 2b — Does a sub-behavioural trace survive overwriting? (DESIGN CARD, APPROVED)

**Status:** APPROVED by Arley 2026-08-11. **v2** after review — see the changelog
at the end; three of the changes would each have invalidated the result silently.

## What changed, and why this is a new question

Phase 2 asked whether ternary models forget less. They do not: on the conflict
pair both arms reach **zero** task-A accuracy. That claim was retracted.

The follow-up asked whether a trace survives below the behavioural floor, and
produced numbers that were entirely an artefact of an unmasked left-padded batch.
Corrected, and analysed with letter fixed effects rather than the ratio:

| arm | retention contrast (A→B minus control) |
| --- | ---: |
| ternary | +0.65 ± 0.36 (~1.8σ) |
| float | +0.57 ± 0.66 (~0.9σ) |

**Same sign in both arms.** So the question is no longer "is ternary special" —
it is **"does a sub-behavioural trace survive overwriting at all, and is it
larger in the ternary twin?"** One seed and 50 keys cannot answer either half.

This is **not** spec §1's H1 or H2. Both are resolved: H1 refuted, H2 retracted.
This is a phase-2 follow-up with its own, narrower claim.

## The card, in the required form

**Hypothesis.** After a conflicting task drives an association to behavioural
extinction (0% accuracy), the original value retains probability mass above what
an identically-trained model that never learned it would assign — and that
surplus may be larger in the ternary twin than in its data-matched float twin.

**Method.** Per arm and seed: train A, and separately train a **placebo A′** whose
value for each key is drawn uniformly from the letters that are *neither* v1 nor
v2; then train the conflicting B from each.

Not a permutation — that was the first draft and it contaminates itself. Over 8
roughly-uniform letters a permutation hands ~1/8 of keys the very letter being
measured (teaching v1 in the control) and another ~1/8 the letter v2 (so those
keys face no conflict during B, breaking extinction matching at training time,
unfixable in analysis). Verified fixed 2026-08-11: the placebo now never equals
v1 or v2 on any key, and shares the conflict key namespace so every key still
genuinely conflicts. The
A→B minus placebo→B contrast isolates "was v1 taught" at matched budget, matched
format exposure, and matched total steps.

**Metric.** A two-way fixed-effects estimate on log-probabilities over the non-v2
letters: `log p = key + letter + γ_v1·1[ℓ=v1] + γ_v'·1[ℓ=v']`, fitted separately
per condition, with the reported quantity the **contrast** γ_v1(A→B) −
γ_v1(placebo→B) in log units.

Both dummies are fitted in **both** conditions, which doubles the evidence for
free: γ_v'(placebo→B) is a second, independent measurement of the same trace (of
v′), while γ_v'(A→B) and γ_v1(placebo→B) are pre-registered **negative controls**
— a letter that was never taught in that condition must show γ ≈ 0.

Linear FE on log-probabilities is the right form here: probabilities come exactly
from the softmax (there is no sampled outcome for a multinomial likelihood to
model), a multiplicative trace is additive in log, and the key fixed effect
already absorbs each key's log normaliser — so conditional logit would be the
same model with extra machinery. Separate per-condition fits differenced is
preferable to one pooled model, because letter marginals genuinely differ between
trained models and pooling would need full condition×letter interactions, which
is identical to fitting separately.

**Seeds.** 3, per spec §7. 200 keys per task.

**This required a code fix, not just a bigger number.** `trace._read` called
`synthetic.make(task, split)` and dropped both `n_keys` and `seed`, so asking for
200 keys silently returned 50 and three "seeds" trained on the *same* seed-0
assignments in a different order — replicating optimizer noise only. Plumbed and
tested 2026-08-11: 200 keys now arrive and 176/200 assignments differ between
generator seeds (25 coincidences expected from 8 letters).

**Estimated GPU-hours: ~4.3 if the measured step times are heat-soaked, ~7 if
they are cold** (cap ~40 either way). The 2.1 s and 0.75 s figures come from
short-sequence runs whose thermal state was not recorded, and CLAUDE.md mandates
budgeting at the 1.9× derate when in doubt — so treat ~7 as the planning number
and record which it turned out to be.

Disk ~35 GB of final weights, **plus** the rotating trainer checkpoints
`sequential.py` keeps (2 per run, weights + optimizer). Nothing deletes those
today, so the run must `rm -rf checkpoint-*` per run after `save_model` or the
transient footprint is several times 35 GB.

## Three things that must be fixed first — all found by review, none optional

1. **The attention mask.** `h2_falsifier.py` left-padded and passed no mask, so
   pad tokens were attended and RoPE positions shifted; 30 of 50 rows were
   contaminated and every headline number was an artefact. **Fixed 2026-08-11.**
   The card additionally requires a **batch-size-invariance assertion at
   runtime**, because that check would have caught this in seconds and costs
   nothing. **Score in fp32 with autocast off**, and pin the tolerance from a
   one-off shakedown rather than asserting 1e-6: under bf16 autocast, batch 1 and
   batch 4 change matmul shapes and reduction order and will *not* agree to 1e-6,
   so the assertion as first written would have failed spuriously and been
   loosened mid-run — which is this project's documented failure mode, not a
   safeguard against it. 24 forward-only passes on a 360M model in fp32 roughly
   doubles a 0.1 GPU-h line item, i.e. noise.
2. **The v1/distractor ratio is retired.** "Never-taught distractors" do not
   exist: each of the 8 letters is a trained B-answer for 2–12 other keys and an
   A-value for 2–13, so only within-key identity separates v1 from a distractor
   and any across-letter mean is contaminated by letter marginals. The FE
   estimator above replaces it. **The ratio must not appear in the output.**
3. **The generator's collision bump.** `synthetic.py:85-87` resolves a v1/v2
   collision by taking the *next* letter, making P(v2 = v1+1 mod 8) 25% rather
   than 12.5%, on adjacent token ids — which plausibly produced the ternary
   B-only γ of +0.67 with A never taught. Resample v2 uniformly from the 7
   non-v1 letters. **Note this changes the synthetic tasks, so phase 2b's numbers
   are not directly comparable to phase 2's** — acceptable, since everything here
   is retrained anyway, but it must be said in the write-up.

## Design

| factor | levels |
| --- | --- |
| arm | ternary twin, float twin |
| history | A (real), A′ (placebo: same keys, value drawn uniformly from VALUES \ {v1, v2}) |
| seed | 0, 1, 2 |

Per (arm, seed): train A (300 steps), train A′ (300), then B from each (300
each) = 4 runs. 24 runs total. **A fresh output root** (`outputs/phase2b`), not
`outputs/phase2`: the generator has changed, and the `COMPLETE`+weights skip
logic plus open item 11 (`content_hash` covers config, not code) would otherwise
silently reuse old-generator checkpoints. lr 1e-4 both arms (item 26), `completion_only`
(item 13), full FT, 8-bit Adam, `max_length=256`, per-arm micro-batch (ternary
4×4, float 2×8 — the float arm OOMs at 4×4), through `sequential.py`.

Checkpoints weights-only; kept until the FE analysis has run, then pruned — and
**not** pruned while any downstream run still branches from them, which broke
the phase-2 chain twice.

## Gates, pre-registered

- **Extinction gate.** Both arms must reach ~0 task-A accuracy after B. If not,
  this is not the sub-behavioural regime and the question does not apply.
- **B-mastery gate.** Each arm must reach p(v2) ≥ 0.99 on the shared prompts —
  a per-arm regime check. The **cross-arm** matched-ratio comparison is
  **report-only**, deliberately: the key fixed effect absorbs each key's total
  non-v2 mass and therefore the B-mastery level, which is precisely the mechanism
  behind the retracted 2-of-6-nats artefact. Gating on a 7× ratio of near-zero
  NLLs that the estimator does not depend on would only set up a mid-run
  renegotiation.
- **Cross-instrument consistency gate.** For every scored checkpoint, −log p(v2)
  from the scoring call must agree with `nll_b_after_B` from the sequential probe
  file within a stated tolerance. This is free — both numbers already exist — and
  it is the check that would have caught the padding bug in seconds: the recorded
  0.9961 was 3.9e-3 nats against the probe's own 2.0e-5, a 200× mismatch waved
  through as "≈1.0".
- **Placebo gate — the load-bearing one.** γ_v1(placebo→B) must satisfy
  **|γ| < 0.3**, an equivalence margin rather than "indistinguishable from 0":
  at SE ~0.15 a plain significance test passes trivially by being underpowered.
  If a letter the model never learned still shows a trace, the estimator is
  contaminated by something other than memory and no contrast is interpretable.
  Same role the base model played in the item-20 gate.
- **Extinction, both conditions.** A must extinguish after B, and A′ must
  extinguish after B in the placebo condition — the symmetric check.
- **Claim thresholds, fixed now:** a trace exists only if the contrast exceeds
  **3 SE** in an arm; it is ternary-specific only if the *difference between
  arms* exceeds **3 SE**.

  **Inference is at the seed level**, not the cell level: the 7 log-probs within
  a key are mechanically dependent and every cell in a run shares one model, so
  cell-level SEs are not credible. The primary estimator is the **paired per-key**
  contrast (within-key, letter-adjusted v1 residual in A→B minus the same in
  placebo→B), which cancels key idiosyncrasies shared across conditions; the
  reported statistic is the mean of the three per-seed contrasts ± sd/√3 on
  **t with 2 df**, where 95% needs 4.3×SE rather than 2.

  **Honest power, both arms.** Scaling the measured one-seed SEs (ternary 0.36,
  float 0.66) by ÷2 for 4× keys and ÷√3 for seeds gives ~0.10 and ~0.19. So the
  measured +0.65 resolves at ~6σ in the ternary arm but +0.57 resolves at only
  **~3.0σ in the float arm — exactly on the threshold, not comfortably**. And the
  arm-*difference* has SE ≈ √(0.10²+0.19²) ≈ 0.21, so the 3-SE gate detects only
  differences ≥ **~0.65**; the observed difference is 0.08. **A null on the
  arm-difference half therefore bounds it above ~0.6 and says nothing tighter** —
  that limitation goes in the write-up, not just here. These scalings also assume
  between-seed variance is zero, which nobody has measured, because one seed
  exists.
- **If the trace is present in both arms with no arm difference**, that is the
  result: sub-behavioural retention is a property of overwriting, not of ternary.
  Report it and stop; do not go looking for a slice where the arms differ.

## Budget

| item | GPU-h |
| --- | ---: |
| ternary: 3 seeds × 4 runs × 300 steps @ ~2.1 s | 2.1 |
| float: 3 seeds × 4 runs × 300 steps @ ~0.75 s | 0.75 |
| scoring (24 checkpoints, 200 keys, forward only) | 0.1 |
| contingency (~50%) | 1.4 |
| **total** | **~4.3** |

Above the ~2 GPU-h the review estimated, because that figure assumed B-only
rather than the matched placebo. The placebo is worth the extra hour: B-only
differs from A→B in total steps *and* format exposure, and for the ternary arm
its extra steps also continue conversion.

## The process fix this card exists to enforce

Six instances of the same failure class, the last two inside the instrument built
to check the previous one. The pattern is **not** "a metric that measures the
wrong object" — it is *writing a second measurement path instead of reusing the
audited one*. `probes.py:355-361` already left-padded and masked correctly.
`sequential.checkpoint_ok` already had the weights-exist check that was
re-derived three times in one day.

So: **phase 2b adds no new scoring path.** It calls `probes`-level primitives, or
it extends them in place with tests. Any new forward pass must carry the
batch-invariance assertion. This is a constraint on the implementation, not a
suggestion.

## What this card does not do

No new hypotheses; H1 and H2 stay resolved. No TRACE tasks (the ternary twin
cannot do them). No escalation ladder. If the placebo gate fails, the answer is
that this cannot be measured with this design — report that and stop, rather than
building a third instrument.


## v1 → v2 changelog (review, 2026-08-11)

1. **`n_keys` and `seed` were not plumbed to the generator.** The card asked for
   200 keys × 3 seeds; the code would have delivered 50 keys and one set of
   assignments. Fixed in `trace._read` with tests. This was the seventh instance
   of the project's failure class, caught before the run rather than after.
2. **The placebo was a permutation and contaminated itself** — ~1/8 of keys would
   have been taught the measured letter. Now drawn from VALUES \ {v1, v2}, with
   a test.
3. **Batch-invariance at 1e-6 was unachievable under bf16 autocast** and would
   have been loosened mid-run. Now fp32 scoring with a tolerance measured in a
   shakedown first.
4. **Added the cross-instrument consistency gate** (scoring-path p(v2) vs the
   harness probe's `nll_b_after_B`) — free, and the single check that would have
   caught the padding bug.
5. **Power stated honestly for both arms**, at the seed level on t with 2 df: the
   float arm resolves at ~3.0σ (on the threshold, not comfortably) and the
   arm-difference gate detects only differences ≥ ~0.65, so a null there bounds
   the difference above ~0.6 and nothing tighter.
6. **Symmetric two-dummy model** — γ_v′ in both conditions gives a second free
   measurement plus two pre-registered negative controls.
7. **Cross-arm B-mastery gate downgraded to report-only**; the key fixed effect
   already absorbs mastery level, so gating on it invited a mid-run renegotiation
   of a threshold the estimator does not use.
8. **Placebo gate given an equivalence margin** (|γ| < 0.3) instead of
   "indistinguishable from 0", which an underpowered test passes trivially.
9. **Fresh output root, explicit checkpoint cleanup, and honest cold/soaked
   budget** (~4.3 h soaked, ~7 h cold).

The estimator itself, the stop rules, and the "no new scoring path" constraint
were reviewed and kept. The constraint is achievable: `probes._next_token_logprobs`
(probes.py:341-363) already left-pads, masks, and returns the full-vocab
log-softmax at the final position — exactly what the FE model needs. It gains an
fp32 option and a batch-invariance wrapper, extended in place with tests.
