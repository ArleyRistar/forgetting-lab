# Instrument MDE pilot — what can each instrument detect, per GPU-hour? (DESIGN CARD, DRAFT)

**Status:** DRAFT — awaiting Arley's approval. Hard rule 1: nothing here runs
before he approves it.

## What this is, and what it is not

**A new line of work.** The ternary/forgetting programme is closed (H1 refuted,
H2 retracted, 2b answered; open-items list cleaned 2026-08-12). This card does
not continue it and claims nothing about ternary models.

It starts from the phase-0 result that mattered most (LAB-NOTES 2026-08-08,
"phase-0 before/after eval"): after a 400-step LoRA SFT of SmolLM2-360M,
**every benchmark delta sat at or under ~1.1 SE** — arc_easy, hellaswag, and
IFEval all statistically silent — while held-out loss moved 1.362 → 1.187 and
token accuracy 66.1% → 69.0% (all measured). The fine-tune plainly worked and
four benchmark accuracies could not see it. That observation became a lab rule;
it has never been quantified.

The question, precisely:

> **For a *paired* fine-tuning comparison at small scale — same base weights,
> before vs after, paired by evaluation item — what is the minimum detectable
> effect (MDE) of each available instrument, and what does it cost in
> GPU-hours to reach it?**

Two lines this card must stay inside:

- **Paired, not cross-model.** Published MDE / sample-complexity work asks
  whether a benchmark separates model A from model B — unpaired,
  leaderboard-shaped. Here both "models" share base weights and the comparison
  is paired per item, which removes item difficulty as a variance source; the
  noise floor is different and lower, and nobody has characterised it at
  360M–1.7B. The phase-0 SE column was explicitly the *unpaired* approximation
  because those runs lacked `--log_samples` (LAB-NOTES caveat, 2026-08-08).
  The flag entered `scripts/eval.sh` afterwards and, as of 2026-08-14, had
  still never been exercised — both phase-0 eval directories hold
  `results_*.json` and nothing else. A two-minute `--limit 2` probe (LAB-NOTES
  2026-08-14) confirmed it now emits pairable per-item records on all three
  tasks ᵐ, so the paired analysis is possible in fact, not merely in principle.
- **Cost-denominated.** The deliverable is not "the spread" — it is **GPU-hours
  per unit of detectable effect**, per instrument, on this box. The measured
  cost asymmetry is the point: one IFEval run costs 1 h 46 m ᵐ while the
  held-out NLL probe costs ~4.2 s/task ᵐ (LAB-NOTES 2026-08-08, probe-cost
  entry) — a ~1500× price gap whose *resolution* gap has never been measured.

**Failure mode to refuse:** drifting into "benchmark variance across
seeds/prompts" in general. That is published (Sclar et al. 2023; the
format-sensitivity and multi-variant reliability audits of 2026). If a revision
of this card no longer needs the word *paired* in its hypothesis, it has become
that literature and should be killed.

This is a PILOT. Its only job is to answer "is the full study worth running"
inside ~14.1 derated GPU-h, and it pre-registers the answer NO as a live
outcome (see the abandon criteria).

## The card, in the required form

**Hypothesis.** In a paired before/after comparison at 360M, (a) pairing by
item shrinks each benchmark's delta SE materially (≥20%) below the unpaired
`sqrt(se_b² + se_a²)` approximation phase 0 had to use, and (b) the available
instruments differ by ≥10× in GPU-hours per unit of minimum detectable effect,
with likelihood probes cheapest per unit of resolution — making a full study
(more scales, more instruments) worth designing. Both halves are falsifiable
here, and either falsification kills the full study (see abandon criteria).

**Method.** Re-run the exact phase-0 smoke recipe (`flab.train`: LoRA r=16,
`smol-smoltalk[:4000]`, effective batch 16 = 4×4 accumulation, max_length 1024,
bf16 + gradient checkpointing, 400 steps — the recipe whose effect size is
already measured) at **seeds 0, 1, 2**. Merge each adapter into a throwaway
copy before any generative eval (`merge_and_unload`; the unmerged path costs
1.89× ᵐ, LAB-NOTES 2026-08-07). Evaluate base once and each merged seed once,
on four instruments, all from paths already wired up — no new dependencies, no
new scoring path (the phase-2b constraint, kept):

1. **Held-out NLL / token accuracy** on the held-out `smol-smoltalk` split,
   per-example, via the existing probe primitives (`probes.py` path).
2. **arc_easy + hellaswag binary accuracy** (`acc`, `acc_norm`) via
   `scripts/eval.sh`, paired per `doc_id` ᵐ.
3. **arc_easy + hellaswag loglikelihood margin** — the continuous per-choice
   logprobs in `resps`, free from the same forward passes ᵐ (LAB-NOTES
   2026-08-14). Added after the probe found them: a binary per-item score can
   only move −1/0/+1, a margin moves continuously, so this is a strictly more
   sensitive read of data the run already produces. Subject to the
   quantisation-floor gate below.
4. **IFEval** (generative) via `scripts/eval.sh`, merged weights only, paired
   at both prompt and instruction level ᵐ.

**Batch size is pinned to a fixed integer, not `auto`** (open item 35). The
auto-detector is allocator-state-dependent — observed OOM-thrashing down to
batch 2 while the run header simultaneously reported 64 ᵐ — and open item 32
makes batch size a scoring variable. Two runs compared per item must be scored
at the same batch size, or the pairing measures the instrument rather than the
model.

Plus two **base-vs-base repeats** as determinism gates, both at the pinned batch
size — if identical weights do not give identical per-item scores, per-item
pairing is contaminated by harness noise, and that component must be measured
and reported rather than assumed away:

- the loglikelihood pair in full (~4 min ᵐ);
- **IFEval at `--limit 100`** (~0.65 h ᶜ). This one is not optional. The design
  uses a single base IFEval run as the shared "before" for all three seeds,
  which is valid only under exact reproducibility, and generative decoding is
  where nondeterminism compounds worst — one flipped token early rewrites the
  entire continuation, where a loglikelihood merely shifts in a low decimal. It
  guards 62% of the pre-contingency budget for 6% of it.

All analysis is CPU-side on the `--log_samples` outputs and probe files:
per-item deltas → paired SE per instrument; the three seeds → between-seed sd
of each instrument's delta; both → MDE and GPU-h-per-MDE.

**Metric.** Per instrument: (i) **paired SE** = sd(per-item deltas)/√n, vs the
unpaired approximation from the same run — the *pairing gain*; (ii)
**between-seed sd** of the aggregate delta (3 seeds, t with 2 df — inference at
the seed level, per the phase-2b convention: items within a run share one
model); (iii) **MDE** = the smallest true effect resolvable at 3 SE under the
larger of the two noise components; (iv) the deliverable, **GPU-hours per
paired comparison at that MDE**, using this card's own measured wall-clocks.
The headline table is instrument × {paired SE, seed sd, cost/run, GPU-h to
detect the smoke-run-sized effect at 3 SE} — where "cannot, at any budget that
fits this box" is an admissible and interesting cell value.

**Seeds.** 3 (0, 1, 2). Seed 0 is deliberately a replication of the phase-0
smoke run and doubles as a calibration gate (below). Three seeds is the spec-§7
floor; the phase-1b lesson (LAB-NOTES 2026-08-09: one seed in three flipped the
sign of BWT; sd ~3× the mean) is exactly why the seed-variance component is
measured rather than assumed here.

**Estimated GPU-hours: ~14.1 thermal-derated** (breakdown below; cap ~40,
target 8–15). Training is budgeted from the measured cold step time at the
mandated 1.9× derate, not from the run average. IFEval is *not* additionally
derated, for two reasons: the 1 h 46 m figure is a measured full-run wall clock
ᵐ (LAB-NOTES 2026-08-07, base SmolLM2-360M at 11.86 s/it) rather than a cold
rate, and IFEval runs at only ~52% GPU utilisation ᵐ (LAB-NOTES 2026-08-08 —
sequential decoding, not compute-bound), so a derate measured on a
compute-saturated training workload does not transfer to it. The 1.9× on
IFEval's rate would add ~6 h and still clear the cap, so nothing here depends
on that judgement being right.

## Design

| factor | levels |
| --- | --- |
| model | SmolLM2-360M (float, bf16) — one scale; the 135M/1.7B sweep is the full study, not the pilot |
| intervention | 400-step LoRA SFT, phase-0 smoke recipe, effect size known ᵐ |
| seed | 0, 1, 2 |
| instrument | held-out per-example NLL; arc_easy+hellaswag binary acc; arc_easy+hellaswag logprob margin; IFEval prompt/inst level — all paired per item |

Runs: 3 training runs; 4 full eval passes (base + 3 merged seeds) on all three
instruments; 1 base-vs-base loglikelihood repeat. Fresh output root
(`outputs/mde-pilot`) — open item 11 (`content_hash` covers config, not code)
makes reusing an old root a silent-corruption risk. Long legs in tmux on the
box, completion markers per chain, one GPU job at a time.

**VRAM, against `peak_reserved`** (the 8–13% reserved-over-allocated gap is
what occupies the card ᵐ, LAB-NOTES 2026-08-08): LoRA training peaked at
1.86 GiB of 7.66 ᵐ (smoke run; the notes do not label reserved vs allocated —
flagged, but the headroom dwarfs the ambiguity); loglikelihood evals measured
4.3 GiB ᵐ; the base-model IFEval ran on this card in phase 0, so the merged
copies (identical architecture and dtype) fit by the same evidence. Everything
sits under the 7.5 GiB usable budget with ≥40% headroom. Disk: 3 adapters
(~35 MB each ᵐ) + 3 merged bf16 copies (~720 MB each ᵐ) + eval logs ≈ 2.5 GB ᶜ;
merged copies deleted after their evals, adapters kept until sign-off (open
item 33).

## Gates, pre-registered

- **Determinism gate.** Both base-vs-base repeats must reproduce per-item
  scores exactly (or to a tolerance measured *before* the result-bearing evals,
  not renegotiated after — the phase-2b lesson). Item alignment between any two
  runs is verified on `doc_hash`/`prompt_hash` ᵐ, never assumed from row order.
  If the harness is not repeat-stable, the repeat component joins the noise
  model and the paired SEs are reported with it. If **IFEval specifically** is
  not repeat-stable, the shared-base design is invalid and IFEval needs its own
  base run per seed — which triples its cost and, at 1 h 46 m a run, means
  dropping IFEval from the pilot rather than paying for it.
- **Quantisation-floor gate** (open item 36). Before the logprob margin counts
  as an instrument, confirm the fine-tuning effect on per-item margins exceeds
  the bf16 step — 0.125 at |LL| 16–32, 0.25 at 32–64 ᵐ (LAB-NOTES 2026-08-14).
  Below the step, a shift is not representable in the log at all, so the margin
  would be an artefact of the logging dtype and only binary accuracy survives.
  Costs nothing: it is a comparison on data the run already produced.
- **Replication gate.** Seed 0 must land near the phase-0 measured endpoints
  (held-out loss ~1.187, token acc ~0.690 ᵐ) within a tolerance set from the
  between-seed sd this very card measures. A gross miss means the recipe or the
  environment drifted since 2026-08-07 and every comparison to the phase-0
  numbers is void — stop and diagnose before spending the IFEval hours.
- **Report the batch size of every logprob measurement** (open item 32 — a
  standing rule; float models are batch-stable in prior measurements, but the
  number is free to record and the rule exists because assuming it once cost a
  rescore).
- **Claim thresholds, fixed now:** an instrument "detects" the smoke-run effect
  only at ≥3 SE under whichever noise component (item-pairing or between-seed)
  is larger; instruments "differ" in cost-per-resolution only if the ratio of
  their GPU-h-per-MDE exceeds 10×. At 3 seeds, seed-level 95% intervals need
  4.3×SE (t, 2 df) — stated so nobody quietly uses 2× later.

### Abandon criteria — what makes the full study NOT worth running

Pre-registered, because a pilot that can only recommend continuing is not a
pilot:

1. **Pairing buys nothing.** If the paired per-item SE is ≥0.8× the unpaired
   approximation on *both* benchmark instruments, the premise that
   distinguishes this from published unpaired-MDE work is false at this scale —
   ABANDON; write the negative note and stop.
2. **Seed variance drowns everything.** If the between-seed sd of the delta
   exceeds the paired item SE by >5× on *every* instrument, then instrument
   choice cannot buy resolution — the only lever is seed replication, the
   cost-ranking question is moot, and the full study collapses to one sentence.
   ABANDON; the sentence still goes in LAB-NOTES.
3. Otherwise — pairing gain ≥20% somewhere, and ≥10× spread in GPU-h-per-MDE —
   the full study (135M/1.7B scales, more instruments, effect-size sweep) gets
   its own card. This pilot never escalates by itself.

## Budget

| item | basis | GPU-h |
| --- | --- | ---: |
| training: 3 × 400 steps @ 9.22 s/step ᶜ (= 4.85 s cold ᵐ × 1.9 derate ᵐ) | measured cold rate, mandated derate | 3.07 |
| IFEval: 4 models × 1 h 46 m ᵐ (base, merged weights) | measured full-run wall clock | 7.07 |
| IFEval determinism repeat: base × 2 @ `--limit 100` | 11.76 s/prompt ᶜ (= 1 h 46 m ᵐ / 541 prompts ᵐ) | 0.65 |
| arc_easy + hellaswag: 5 passes × ~4 min ᵐ (incl. determinism repeat) | measured | 0.33 |
| held-out NLL probes: 4 models, ~200 examples each | measured ~4.2 s/task ᵐ, rounded up hard | 0.05 |
| merging + model loads | computed, generous | 0.10 |
| contingency (~25%) | — | 2.82 |
| **total** | | **~14.1** |

ᵐ = measured, LAB-NOTES entry cited in the text above. ᶜ = computed —
arithmetic, not a result (hard rule 3); this card's own wall-clocks replace
every one of these numbers once it runs.

Comfortably under the ~40 h cap, inside the 8–15 h pilot target. The single
biggest line is IFEval at 62% of the pre-contingency total (7.07 of 11.27 ᶜ) —
which is itself the point of the study: if the most expensive instrument also
has the worst MDE, that is the headline table row.

## What this card does not do

No ternary models, no forgetting claims, no reopening of H1/H2. No new eval
tasks, dependencies, or scoring paths — `scripts/eval.sh` and the `probes`
primitives as they stand. No 135M or 1.7B runs (full study, own card). No
prompt-format or few-shot variations — that is the published territory this
card exists to stay out of. Per spec §9, the novelty search ("paired
before/after MDE at small scale") gets re-run before the *full* study's card is
drafted; this pilot's spend is justified by the cost question alone, which is
box-specific and cannot be scooped.
