# Phase 1e — Can the converted twin still LEARN? (DESIGN CARD, awaiting approval)

**Status:** DRAFT — not approved, no compute spent. Spec §3 gate.

## Why this card exists

Phase 1d's item-20 gate showed the ternary twin discriminates on **no**
pre-existing task (best 0.8 SE, against base's 6.1 and 9.4). I concluded phase 2
was blocked. That conclusion does not follow, and the contradiction is in my own
write-up: the same entry says the synthetic tasks' gate failure "is not evidence
they are unusable", because those associations are *taught* by the experiment
rather than retained from pretraining.

**The gate answered "can it recall?". Phase 2 needs "can it learn?".** Arley's
decision on item 31 scoped forgetting to what the converted model *relearned*, so
the second question is the one that governs. Nobody has asked it.

If the ternary twin can learn a fact set, phase 2 runs as **teach A → teach B →
measure forgetting of A**, with no new conversion and no distillation. If it
cannot, that is a *stronger* result than the current one — "ternarisation at 66M
tokens destroys plasticity, not merely recall" — and it justifies the ~25 GPU-h
distillation route on evidence rather than assumption.

Either outcome is worth ~1 GPU-h. Neither is currently known.

## The card, in the required form

**Hypothesis.** The converted ternary twin can still acquire new key→value
associations from fine-tuning, despite having lost measurable recall of every
pretrained task tested.

**Method.** Full fine-tune each twin on `synth-conflict-a` (50 nonsense keys,
single-letter values, 4 repeats), prompt masked (item 13's `completion_only`),
lr 1e-4 (item 26's same-LR-both-arms decision), then re-score the held-out split.
Score both twins **before** training as well, which must land at chance since the
associations are invented for this project.

**Metrics.**
1. *Primary:* held-out answer NLL against the **analytic chance level**,
   `log(8) = 2.0794` nats. `synthetic.py` was built so one answer token carries
   the whole association, precisely so this number exists without a control.
2. *Confirmatory:* the same paired-derangement delta the item-20 gate used. A
   model can drive NLL below chance by learning the answer *marginal* ("answers
   are single letters A–H") without learning any mapping; the derangement
   isolates the association, because it holds the answer distribution fixed and
   breaks only the pairing.

Both are needed: metric 1 alone can be passed without learning anything.

**Seeds.** One. This is a go/no-go feasibility question, not an effect size.
Spec §7's ≥3 seeds binds phase 2.

**Estimated GPU-hours: ~1.0** (cap ~40). Breakdown below.

## Decision rule, pre-registered

| ternary twin | float twin | verdict |
| --- | --- | --- |
| learns | learns | **Phase 2 proceeds on taught tasks.** Write the phase-2 card; no new conversion needed. |
| does not learn | learns | **Plasticity is gone, not just recall.** Stronger negative result than 1d's. Distillation (item 17) becomes the evidence-backed route. |
| does not learn | **does not learn** | **The check is uninformative** — task, harness or hyperparameters are at fault, not the twin. Investigate; conclude nothing about ternary. |

"Learns" means held-out answer NLL **below 1.4 nats** (a third under chance) *and*
derangement delta **≥3 SE**. Both stated now so neither can be adjudicated after
seeing the numbers.

**The float twin is the positive control and does the same work the base model
did in the item-20 gate.** Without it, "the ternary twin did not learn" cannot be
distinguished from "nothing could have learned this", and phase 1d already showed
how easily that ambiguity arises.

## Work

### Task 1 — run it through `sequential.py`, not a fresh script

The phase-2 harness has never trained a ternary model. `_load_base` was routed
through `flab.loading` on 2026-08-10 (item 21) and unit-tested, but never
exercised end to end. Using it here validates the phase-2 path on a 1 GPU-h job
instead of discovering its faults inside a 7-run experiment.

Assertions to add or confirm in the run:
- `assert_ternary` before and after training, so a run that silently
  de-ternarised cannot be reported;
- λ == 1 at every step (the null arm's `AssertLambdaOne` pattern);
- `completion_only` genuinely masks the prompt — verify on one encoded example
  that only answer tokens carry labels, rather than trusting the flag.

Fall back to a standalone script only if the harness fights this, and record the
reason if so.

### Task 2 — score before and after, both twins

Before: both twins must sit near 2.0794. A twin already below chance on invented
keys would mean the eval leaks into training, and the check would be void.

After: primary and confirmatory metrics above.

### Task 3 — write up, and state the phase-2 consequence explicitly

LAB-NOTES entry with the four numbers (two twins × before/after), the derangement
deltas, and which row of the decision table fired.

## Budget

| item | GPU-h |
| --- | ---: |
| ternary fine-tune, ~300 steps (short sequences) | 0.3 |
| float fine-tune, ~300 steps | 0.2 |
| four scoring passes (2 twins × before/after) | 0.1 |
| contingency (~80%, step time on short sequences is unmeasured) | 0.4 |
| **total** | **~1.0** |

Contingency is deliberately generous: every measured step time in this project
is at `seq_len=1024`, and these prompts are one line. The step time could be
several times faster, or the packing could behave unexpectedly — this is the
first job in the project at short sequence length.

Disk: negligible (a handful of checkpoints).

## What this card deliberately does not do

Not phase 2: no task *sequence*, no forgetting measurement, no seeds, no
escalation ladder. It answers one question — can the twin learn at all — and
stops. Phase 2 needs its own card whichever way this lands.

It also does not revisit the item-20 gate. That verdict stands and was
independently checked; this asks a different question.
