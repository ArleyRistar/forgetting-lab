# Phase 1b — Calibration Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish that the phase-1a instrument produces trustworthy numbers,
by (a) replicating the core protocol of arXiv 2606.27634 and matching its
trends, and (b) building one synthetic forgetting control whose answer is known
analytically. Spec §6 1b. Nothing result-bearing runs until this passes.

**Architecture:** Almost no new machinery. The harness, probe, run state and
supervisor all exist; 1b mostly *parameterises* what phase 1a hardcoded, adds a
reference-set drift probe, and adds the continual-learning metrics the paper
reports so the two can be compared at all.

**Precondition:** phase 1a complete (it is, `408ff61`).

## The paper, verified 2026-08-09 (not assumed)

[arXiv 2606.27634](https://arxiv.org/abs/2606.27634) — *Continual Learning for
Sequential Personalization of Small Language Models: A Stability Monitoring
Analysis*, Paula, Kupssinskü & Barros. Fetched and read; every number below is
quoted from it rather than inferred.

| | their protocol |
| --- | --- |
| models | Qwen 3.5 **0.8B**, **Llama 3.2 1B Instruct**, Gemma 3 1B IT |
| tasks | **FOMC → ScienceQA → NumGLUE-cm**, 500 train each, plus reversed |
| LoRA | r=8, α=16, dropout 0.05, no bias, **target `all-linear`** |
| optim | AdamW, **reset per task**, lr 5e-5 |
| batch | 2 × grad-accum 8 = **16 effective** |
| schedule | **1 epoch per task** (≈31 steps at 500 examples) |
| seq len | **512** |
| reference set | fixed, disjoint from every task's train *and* eval |
| CL metrics | ACC, BWT (`a_k,j − a_j,j`), FWT |
| stability | **KL from base**, entropy change ΔH, top-2 margin |

Their headline results, which are what we must reproduce the *shape* of:

- Qwen most stable — final accuracy **0.591 ± 0.012**, KL drift **0.300**.
- Gemma unstable — final **0.320 ± 0.029**, KL peak **1.623 ± 0.157**.
- **KL vs accuracy correlate negatively: r = −0.497, p < 0.001**, across models.
- **KL ≈ 0.8** proposed as an instability threshold, independent of task order.

That last pair is the real calibration target. Matching an absolute accuracy on
a different model would prove nothing; reproducing *the negative KL–accuracy
relationship* is a claim about the phenomenon.

## What this forces us to change

Six mismatches between their protocol and what phase 1a built. None are hard,
but they must be deliberate rather than discovered mid-run.

| # | mismatch | ours (1a) | theirs |
| --- | --- | --- | --- |
| 1 | TRACE variant | `_5000` (pinned) | **`_500`** — 500 train/task |
| 2 | task order | FOMC → Py150 → ScienceQA | **FOMC → ScienceQA → NumGLUE-cm** |
| 3 | LoRA | r=16, α=32, 7 named modules | **r=8, α=16, `all-linear`** |
| 4 | schedule | fixed `max_steps` | **1 epoch** |
| 5 | seq len / lr | 1024 / 2e-4 | **512 / 5e-5** |
| 6 | metrics | NLL + token acc | **ACC/BWT/FWT + KL/ΔH/margin** |

Mismatch 6 is the interesting one and is discussed under task 4.

## Two data traps, found while checking (2026-08-09)

**Lima's `eval` and `test` splits have 100% empty answers** — all 300 rows in
each, in both `_500` and `_5000`. Only `Lima/train` carries real answers
(median 1563 chars). Lima is the obvious reference-set candidate (it is TRACE's
replay set, disjoint from every task), and reaching for `Lima/eval.json` — the
natural choice — yields a set with **zero scorable answer tokens**. Use
**`Lima/train`**, held out and never trained on.

Encouragingly, `probes.py` already catches this: an example with an empty
answer contributes no unmasked labels, `n_tokens` reaches 0, and the probe
returns `warning: "zero answer tokens scored; the NLL below is not a
measurement"` rather than a plausible-looking number. Task 2 turns that into a
regression test — it is the exact failure the warning field was built for.

**NumGLUE-cm has only 41 eval examples** (and 81 test). It is their *third*
task, so it carries the most forgetting signal, on the smallest held-out set of
any TRACE task. Report its `n` beside every number it produces; `n_eval` is
already clamped and reported, so this needs discipline, not code.

---

### Task 1: Parameterise what 1a hardcoded

**Files:** modify `src/flab/trace.py`, `src/flab/runconfig.py`,
`src/flab/sequential.py`; add `configs/calib-paper.yaml`; extend tests.

- [ ] **Step 1: Make the TRACE variant a config field**

`trace.VARIANT` is pinned to `_5000`. Replication needs `_500`. Make it a
parameter threaded from `RunConfig` (default `_5000`), and keep the guard that
`_500`'s 20Minuten lacks `eval.json` — pin, never auto-discover.

- [ ] **Step 2: LoRA hyperparameters into the config, `all-linear` supported**

`r`, `alpha`, `dropout`, `bias`, `target_modules` are currently constants in
`sequential.py`. Move them to a `lora:` block. `target_modules: all-linear` must
map to peft's `"all-linear"`, which is not the same set as our 7 named modules —
it includes the LM head's projections on some architectures, and that changes
what is being adapted.

- [ ] **Step 3: `epochs` as an alternative to `max_steps`**

Stages take `max_steps`. The paper trains **1 epoch**. Accept either, reject
both-or-neither at validation time, and record the resolved step count in run
state so the two are comparable afterwards.

- [ ] **Step 4: Confirm the optimizer really is reset per task**

The paper resets AdamW per task. Our harness builds a fresh `SFTTrainer` per
stage, which *should* mean a fresh optimizer — but "should" is not "does". Assert
it: check optimizer state is empty at the first step of stage k>0. If it is not,
the comparison is invalid in a way no output would reveal.

- [ ] **Step 5: `configs/calib-paper.yaml` reproducing their protocol exactly**

Every value from the table above. Commit it as the executable record of what we
replicated.

---

### Task 2: The reference set

**Files:** modify `src/flab/trace.py`, `src/flab/probes.py`; add tests.

- [ ] **Step 1: Load `Lima/train` as a reference set**

Held out, never trained on, disjoint from all task data — satisfying the
paper's `R ∩ D_train = ∅` and `R ∩ D_eval = ∅`. Fixed selection by seed, as
with every other probe set.

- [ ] **Step 2: Regression-test the empty-answer trap**

Assert that probing `Lima/eval` (all-empty answers) returns `n_tokens == 0` and
a non-null `warning`, and **not** a number. This is the safety net working on a
real case rather than a synthetic one; it is worth pinning permanently.

- [ ] **Step 3: Probe the reference set at every boundary**

Same cadence as the task probes. Cheap — the whole 4-boundary probe cost 84.6 s
in the shakedown.

---

### Task 3: Stability metrics — KL, entropy, margin

**Files:** modify `src/flab/probes.py`; add tests.

This is the only genuinely new measurement in 1b, and it is the one that
matters most downstream: **KL from the base model on a fixed reference set is a
distributional drift measure**, which is conceptually the float-side analogue of
phase 1d's ternary flip-fraction. Building it here means phase 2 compares
like with like.

- [ ] **Step 1: KL divergence from the base model**

Per reference-set token, `KL(p_base ‖ p_current)` over the full vocabulary,
averaged. Needs the base model's logits, so either keep a frozen copy (memory)
or — cheaper and exact under LoRA — **disable the adapter** and run the same
forward pass. `peft` supports `with model.disable_adapter():`. Prefer that: it
is the same weights by construction, and it costs no extra VRAM.

- [ ] **Step 2: Entropy change and top-2 margin**

ΔH against the base distribution, and the log-probability gap between the top
two predictions. Both fall out of the same forward pass as the KL.

- [ ] **Step 3: Validate KL against a known-zero case**

`KL(base ‖ base)` must be exactly 0 with the adapter disabled at stage 0. If it
is not, the two forward passes are not comparable and every drift number after
it is noise. Cheap, decisive, and easy to skip.

---

### Task 4: The metrics bridge — ACC/BWT/FWT

**Files:** add `src/flab/clmetrics.py`; extend `scripts/loss_matrix.py`.

- [ ] **Step 1: Implement ACC, BWT, FWT over the existing matrix**

BWT is `a_k,j − a_j,j` — final performance on task *j* minus performance right
after learning *j*. Our loss matrix already contains exactly this structure; it
has simply been read as NLL rather than accuracy.

- [ ] **Step 2: Compute them on BOTH accuracy and NLL, and report both**

Here is the tension worth stating plainly. The paper's metrics are
**accuracy-based**. Phase 0 and the 1a shakedown both found accuracy to be a
poor instrument at this scale — the sharpest case being ScienceQA moving
+0.1101 NLL while its token accuracy went 0.620 → 0.621, i.e. *nothing*.

We cannot resolve that by picking a side. To compare with the paper we must
compute their accuracy-based metrics; to trust our own results we keep NLL
primary. So compute both, report both, and **treat any disagreement between
them as a finding rather than a nuisance** — a systematic divergence would
itself be worth writing up, since it would say published accuracy-based CL
metrics understate forgetting at small scale.

- [ ] **Step 3: Reproduce the KL–accuracy correlation**

Their central quantitative claim is **r = −0.497, p < 0.001** between KL drift
and accuracy. Compute the same correlation over our boundaries. Matching the
*sign and rough magnitude* is the calibration gate; matching r to two decimals
on a different model would be luck, not validation.

---

### Task 5: The synthetic control with an analytically known answer

Spec §6 1b asks for this separately from the replication, and it does different
work: the replication says "we agree with the literature", the control says
"our instrument reads correctly when we already know the answer".

**Files:** add `src/flab/synthetic.py`, `tests/test_synthetic.py`,
`configs/synthetic-control.yaml`.

- [ ] **Step 1: Two paired synthetic tasks**

Generate key→value associations over nonsense tokens:

- **Conflicting pair.** Task A maps `k → v₁`; task B maps *the same* `k → v₂`.
  Learning B to convergence **must** destroy A — not "probably", but as a matter
  of logic, since one distribution cannot concentrate on both. A's NLL is
  therefore bounded below by roughly `−log P(v₁)` under B's learned
  distribution. **Forgetting here is the analytically known answer.**
- **Disjoint pair.** Task A maps `k_A → v_A`, task B maps a disjoint `k_B → v_B`.
  Nothing forces interference; a model with sufficient capacity can hold both.
  **Zero forgetting is the analytically known answer.**

- [ ] **Step 2: The gate**

The instrument passes if it reports large forgetting on the conflicting pair and
near-zero on the disjoint pair. Failing the first means the probe cannot see
forgetting that is *provably* present. Failing the second is worse — it means
the harness manufactures forgetting that logic says need not occur, and every
number in phase 2 would be suspect.

- [ ] **Step 3: Report the gap, not just the two numbers**

The disjoint arm's non-zero residual is the harness's **noise floor for
forgetting**. Phase 2 must not report effects smaller than it. This number does
not exist yet and nothing downstream is interpretable without it.

---

### Task 6: Run the calibration

- [ ] **Step 1: Choose the model — a real decision, see below**

- [ ] **Step 2: Both task orders**

Forward and reversed, as they did. Their KL ≈ 0.8 threshold is claimed to be
order-independent; that is a testable prediction, and cheap here.

- [ ] **Step 3: ≥3 seeds**

Their accuracies carry ± figures (0.591 ± 0.012), so ours must too. The 1a
shakedown deliberately ran one seed and could not separate effect from variance;
that is acceptable for a shakedown and not for a gate.

- [ ] **Step 4: Write the calibration note (blog #1 material — Arley writes it)**

Claude produces the tables and plots. Per hard rule 2, the prose is Arley's.

- [ ] **Step 5: Record the verdict in LAB-NOTES, including if it fails**

A failed calibration is a finding and stops phase 2 until understood. Write it
up with the same care as a pass.

---

## Decisions needed from Arley

1. **Which model to calibrate on?** Recommendation: **Llama 3.2 1B Instruct** —
   one of their three, and a 1B model fits comfortably under LoRA at 2 B/param
   (spec §4). That makes the comparison *numeric* rather than a trend analogy,
   which is much stronger evidence. Then run SmolLM2-360M as a second arm to
   confirm the trends survive down to the size phase 2 actually uses.

   The alternative — calibrate only on SmolLM2-360M — is cheaper but weaker, and
   has a specific hazard: **their models are instruct-tuned and ours is a base
   model.** The shakedown showed FOMC accuracy starting at exactly 0.000 because
   a base model never emits a bare answer letter. Accuracy-based CL metrics
   computed from a 0.000 starting point are not comparable to theirs in any
   meaningful way. I would not skip the 1B arm.

2. **Design card?** These are the first runs producing numbers we would report,
   so by the rule I wrote into the 1a plan, **this needs a card** — unlike the
   shakedown. Estimated **~6–10 derated GPU-h** for both orders × 3 seeds × 2
   models, well inside the 40 GPU-h cap. Flagging rather than assuming.

3. **Still open from 1a, now more pressing:** the forgetting normalisation
   question (open item 9). Task 4 computes BWT, which is an *absolute*
   difference — so adopting the paper's metric implicitly answers it one way.
   Worth deciding deliberately rather than by inheritance.
