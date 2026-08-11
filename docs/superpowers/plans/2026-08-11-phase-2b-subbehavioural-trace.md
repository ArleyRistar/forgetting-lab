# Phase 2b — Does a sub-behavioural trace survive overwriting? (DESIGN CARD, awaiting approval)

**Status:** DRAFT — not approved, no compute spent. Spec §3 gate.

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

**Method.** Per arm and seed: train A, and separately train a **placebo A′** with
permuted values over the same keys; then train the conflicting B from each. The
A→B minus placebo→B contrast isolates "was v1 taught" at matched budget, matched
format exposure, and matched total steps.

**Metric.** A two-way fixed-effects estimate on log-probabilities over the non-v2
letters: `log p = key + letter + γ·1[letter = v1]`, and the reported quantity is
the **contrast** γ(A→B) − γ(placebo→B), in log units with its SE.

**Seeds.** 3, per spec §7. 200 keys per task (the generator takes `n_keys`; 50
was its default, not a constraint).

**Estimated GPU-hours: ~4.3** (cap ~40). Disk ~35 GB.

## Three things that must be fixed first — all found by review, none optional

1. **The attention mask.** `h2_falsifier.py` left-padded and passed no mask, so
   pad tokens were attended and RoPE positions shifted; 30 of 50 rows were
   contaminated and every headline number was an artefact. **Fixed 2026-08-11.**
   The card additionally requires a **batch-size-invariance assertion at
   runtime** — batch 1 and batch 4 must agree to 1e-6 — because that check would
   have caught this in seconds and costs nothing.
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
| history | A (real), A′ (placebo: same keys, permuted values) |
| seed | 0, 1, 2 |

Per (arm, seed): train A (300 steps), train A′ (300), then B from each (300
each) = 4 runs. 24 runs total. lr 1e-4 both arms (item 26), `completion_only`
(item 13), full FT, 8-bit Adam, `max_length=256`, per-arm micro-batch (ternary
4×4, float 2×8 — the float arm OOMs at 4×4), through `sequential.py`.

Checkpoints weights-only; kept until the FE analysis has run, then pruned — and
**not** pruned while any downstream run still branches from them, which broke
the phase-2 chain twice.

## Gates, pre-registered

- **Extinction gate.** Both arms must reach ~0 task-A accuracy after B. If not,
  this is not the sub-behavioural regime and the question does not apply.
- **B-mastery gate.** Both arms must reach p(v2) ≥ 0.99 on the shared prompts,
  checked on the **ratio** not the rounded value (0.000532 vs 0.000075 was called
  "identical" once already; `claims.check_matched_capability` now enforces this).
- **Placebo gate — the load-bearing one.** γ(placebo→B) must be indistinguishable
  from 0. If the placebo shows a trace, the estimator is still contaminated by
  something other than memory and no contrast is interpretable. This is the same
  role the base model played in the item-20 gate.
- **Claim thresholds, fixed now:** a trace exists only if the contrast exceeds
  **3 SE** in an arm; it is ternary-specific only if the *difference between
  arms* exceeds **3 SE**. At 200 keys × 3 seeds the contrast SE is ~0.10, so the
  measured +0.65 would resolve at ~6σ and a null would be a real null.
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
