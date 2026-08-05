# Forgetting Lab — Design Spec (2026-08-05)

Independent hobby research programme: **catastrophic forgetting in ternary
(1.58-bit) LLMs**. Runs on a spare laptop; Claude implements, Arley directs.

Lineage: v2 draft + adversarial review (Opus 5, 2026-08-05) are in the session
scratchpad; this spec incorporates all accepted review findings. Three earlier
candidate questions were killed by the literature search (see §10).

## 1. Research question

**Does extreme (ternary) quantization change catastrophic-forgetting dynamics
in language models — and is ternary weight-flip behaviour a better predictor of
forgetting than parameter distance is in float models?**

Why this is open: no published work studies forgetting or continual learning in
ternary/BitNet-style models (verified by adversarial literature review,
2026-08-05). The nearest results are one rung up the bit ladder — INT8
quantization improving continual learning via "implicit regularization"
([2512.18934](https://arxiv.org/abs/2512.18934)) — and ternary *pre-training*
transitions ([2502.11895](https://arxiv.org/abs/2502.11895)). So the headline
"low-bit forgets less" is taken; the open contribution is the
**ternary-specific mechanism and its measurement**.

Hypotheses, in priority order:

- **H1 (primary):** the fraction of flipped ternary weights predicts forgetting
  better than parameter-distance metrics predict it in a size- and data-matched
  float twin. Scale-invariant, cheap to instrument, and robust to the known
  QAT-oscillation objection ([Nagel et al. 2022](https://arxiv.org/abs/2203.11086)).
- **H2 (secondary):** at matched capability, ternary models show *different*
  forgetting — possibly less at low learning rates (latent-weight hysteresis),
  possibly burstier (step-like drops when threshold crossings cascade). Framed
  as measurement, not advocacy; a clean no-difference result is still a finding.
  Must engage the oscillation literature: flip *persistence* (does a flip stick?)
  is measured separately from flip *count*.

## 2. Context & constraints

- Researcher: Arley — senior software engineer, solid ML concepts, no hands-on
  NN training. ~5–10 h/week of attention; the GPU can grind ~40–60 unattended
  h/week.
- Success = deep hands-on understanding + real results. Publication (CoLLAs /
  workshop / arXiv) is a bonus, not a requirement.
- Lab box: MSI GS66 12UGS — i7-12700H, 32 GB RAM, RTX 3070 Ti Laptop
  **8 GB VRAM**, 1 TB NVMe. Headless.
- Assistant: Claude Code on the lab box; ~half a weekly Claude subscription's
  tokens available.

## 3. Roles & the design card

- **Claude implements**: environment, harness, experiments, plots, run
  babysitting, literature triage, analysis drafts.
- **Arley directs**: approves every experiment before compute is spent, reviews
  results weekly, writes all conclusions and blog posts in his own words (this
  is where the learning happens; doubles as English-precision practice).
- **Design card gate**: no experiment runs without a one-paragraph card Arley
  has approved — hypothesis, method, metric, seeds, estimated GPU-hours.
  **Hard rule: no single card may exceed ~40 GPU-h** (thermal-derated hours,
  not nominal). Cards accumulate into the lab notebook.

## 4. Hardware envelope

What the 8 GB card supports (bf16, gradient checkpointing, 8-bit optimizer):

- Float: full fine-tune ≤360M; LoRA ≤2B at seq 512–1024.
- Ternary QAT: costs ~1.3–1.5× float training per parameter (latent weights +
  materialised quantized tensors) and **has no PEFT escape hatch** — full
  fine-tuning only. Local ternary ceiling ≈ **350–560M**.
- Pretraining a ~100–135M model from scratch on a few billion tokens: feasible;
  order 50–100 unattended GPU-h per model.
- Evaluation: lm-evaluation-harness ≤2B bf16.

If an approved experiment needs more than this, rent an hourly cloud GPU for
that experiment. No hardware purchases.

Thermals: power-cap ~15% (`nvidia-smi -pl`), elevate chassis, headless. The
resulting ~15–30% throughput penalty is folded into every card's GPU-h estimate.

## 5. Phase 0 — bring-up (~week 1)

Fedora, NVIDIA drivers (RPM Fusion), LAN SSH, `uv` env, Claude Code on the box.
(Tailscale and other remote-access niceties: only when actually wanted.)

Smoke test: LoRA fine-tune SmolLM2-360M on one instruction dataset;
lm-evaluation-harness before/after; one guided walkthrough of the training
script so Arley can explain it end to end.

Deliverable: one reproducible "hello training" run.

## 6. Phase 1 — build the instrument (weeks 2–10)

Everything in this phase exists to make the §1 question answerable.

**1a. Harness.** Sequential fine-tuning (base → task A → task B → …) with
evaluation at stage boundaries, checkpoint saving mid-stage (weights to disk —
*not* evaluated speculatively), and **crash-resume + auto-retry from day one**
(at 5–10 h/week of human attention, a 3 a.m. OOM must cost minutes, not a
calendar day). Development runs: one model (SmolLM2-360M), seed 0. Task data:
TRACE's released datasets (one format, one loader).

**1b. Calibration gate.** Replicate the core protocol of
[2606.27634](https://arxiv.org/abs/2606.27634) (*Sequential Personalization of
Small Language Models*, June 2026 — sequential LoRA on ~1B instruct models):
the only published small-model sequential-FT protocol that fits this card.
Our trends must match theirs before any result-bearing run. Also: one synthetic
forgetting control with an analytically known answer.

**1c. Matched-pair pretraining.** Pretrain from scratch at ~100–135M:
one ternary model (BitNet b1.58-style QAT) and one float twin — same data,
tokenizer, architecture and schedule. This is required because **no public
ternary checkpoint releases its QAT latent weights** (Spectra ships only
ternarized values; Falcon-E re-initialises latents to centroids), and the
latent-weight trajectory is exactly what H1/H2 measure. Save full latent
checkpoints throughout. Budget: ~100–150 unattended GPU-h for the pair.

**1d. Instrumentation.** Per layer, per logging interval: ternary flip
fraction, flip persistence, latent distance-to-threshold histograms; for the
float twin: L2/cosine parameter distance. Note: ternarization uses a per-step
absmean scale, so thresholds move — a weight can flip without moving.
Instrument accordingly (flips are defined by effective-value change, not
latent-value change).

**Eval design for ~100M models:** benchmark accuracies (MMLU/GSM8K/IFEval) are
at floor at this scale and would masquerade as "no forgetting". Use
likelihood-based probes — held-out-fact NLL, per-token loss on task data — plus
chance-adjusted accuracy metrics (framework:
[2510.17776](https://arxiv.org/abs/2510.17776)) where accuracy is used at all.

Deliverables: rig re-runnable from a commit hash; calibration note (blog #1);
the matched model pair with latent-weight history.

## 7. Phase 2 — the experiment (months ~3–6)

Sequential fine-tuning of the matched pair on a task sequence a ~100M model can
actually learn (domain-shift LM corpora + synthetic fact-recall probes; final
design via design cards). ≥3 seeds on all result-bearing runs; conclusions from
paired comparisons only.

- **Primary analysis (H1):** predictive power of flip-fraction/persistence for
  per-task forgetting in the ternary model vs. parameter-distance predictors in
  the float twin — per layer and global.
- **Secondary (H2):** forgetting-curve shape comparison at matched capability —
  magnitude, burstiness, learning-rate sensitivity.
- **Confirmatory arm:** repeat the behavioural (not latent) measurements on
  Spectra TriLMs 99M–560M and/or Falcon-E, with the centroid-initialisation
  caveat documented.

Outputs: a write-up Arley authors; venue optional (CoLLAs 2027 or a NeurIPS/ICML
workshop if the result earns it); blog #2/#3. Honest fallback if the question
fizzles: a replication-and-measurement blog series — acceptable given the
primary goal is learning.

## 8. Cadence & assistant usage

Weekly loop: Arley approves cards + reads triaged papers (~2–3 h); Claude
implements and queues runs midweek; weekend results review (~2–3 h); Arley
writes the running lab notes. Claude tokens: implementation, literature triage,
run babysitting, analysis drafts, edit passes on Arley's prose.

## 9. Risks

- **QAT recipe fiddliness** — ternary pretraining at small scale is
  reproducible (BitNet recipes, Spectra paper) but hyperparameter-sensitive.
  Mitigation: pilot at 50M before the real pair; budget one restart.
- **Pair quality too low to measure forgetting** — mitigation: NLL-probe eval
  design (§6), token-budget check at pilot stage.
- **Oscillation confound** (near-threshold weights flip noisily) — mitigation:
  flip persistence as a first-class metric, H1 framed to survive it.
- **Eval as hidden compute cost** — mitigation: 40 GPU-h card cap; mid-stage
  checkpoints saved but not evaluated by default.
- **Rubber-stamping drift** — mitigation: design-card gate; Arley writes all
  conclusions.
- **Getting scooped** — the niche is empty today (2026-08-05); re-run the
  novelty search before phase 2 begins and before any write-up.
- **Thermals / laptop attrition** — power cap, elevation; weekly `nvidia-smi`
  health check in the run babysitter.

## 10. Non-goals & killed alternatives

- **Killed by literature review (2026-08-05):** capability-type forgetting
  decomposition ([2308.08747](https://arxiv.org/abs/2308.08747),
  [2510.17776](https://arxiv.org/abs/2510.17776)); stability gap in sequential
  fine-tuning ([2606.27634](https://arxiv.org/abs/2606.27634) — now our
  calibration target); minimal replay-ratio curves
  ([2508.01908](https://arxiv.org/abs/2508.01908),
  [2407.17467](https://arxiv.org/abs/2407.17467)).
- No new CL method as a first project; measurement first.
- No ternary work >560M locally; no inference-efficiency/bitnet.cpp work.
- No vision CL; no local chat-assistant hosting; no hardware purchases; no
  production-grade tooling polish.

## 11. Direction review — OPEN, blocks phase 1c (raised 2026-08-05)

Arley's objection: in common practice, low-bit models are produced *from*
trained full-precision models, so studying forgetting "in the ternary model"
may be studying an artifact rather than a real training regime.

Fact status (verified 2026-08-05): at 4–8 bits the objection is exactly right
(GPTQ/AWQ-style PTQ). At 1.58 bits it inverts — ternary PTQ collapses quality,
so the flagship ternary models (BitNet b1.58-2B4T, Spectra TriLMs,
[Falcon-E](https://falcon-lm.github.io/blog/falcon-edge/)) are trained from
scratch with QAT. The conversion route that does exist
([HF Llama3-8B→1.58bit](https://huggingface.co/blog/1_58_llm_extreme_quantization),
[2502.11895](https://arxiv.org/abs/2502.11895)) is itself continued QAT
training on 10–100B tokens, not a post-hoc transform. Either way, sequential
updates to a ternary model happen in QAT space — the object of study exists on
every route to ternary.

What survives of the objection (kernel to settle before phase 1c spends
compute):

- **Deployment realism.** Who sequentially updates ternary models in practice —
  and does the motivation section survive the fact that QAT fine-tuning needs
  full float latents (weakening the "on-device learning" story)?
- **Route realism.** If conversion-from-float becomes the dominant production
  route, a from-scratch pair studies the less-real object.

Routes on the table:

- **Route A (current §6 plan):** pretrain a matched ~100M ternary/float pair
  from scratch. Mirrors how flagship ternary models are made; cleanest control;
  ~100–150 GPU-h before any experiment; evals near floor at 100M.
- **Route B (from the objection):** take a public float model (e.g.
  SmolLM2-360M) and ternarize it via continued QAT. The float weights *are*
  the initial latents, so we own the full latent trajectory without
  pretraining; capabilities (and evals) far above floor; directly studies the
  conversion route. Confound: the conversion quality gap must be measured and
  argued around.
- **Route C:** B first as a cheap pilot (doubles as the QAT-recipe shakedown),
  A only if B shows signal — and then route-to-ternary itself becomes a
  studied variable (does provenance change forgetting?).

To verify: whether Falcon-E's released bf16 revision is the true QAT master
weights from its single-run paradigm (if yes, the "no public latents" premise
weakens and a rented-GPU Falcon-E-1B arm becomes an alternative to Route A).

## 12. Open questions (resolved via design cards, not now)

1. Pretraining corpus and token budget for the 100–135M pair (candidate:
   FineWeb-Edu sample; budget set after the 50M pilot).
2. QAT codebase: adapt a BitNet b1.58 reference recipe vs. Spectra's
   ternarization — decide in phase 1a after reading both.
3. Task-sequence design learnable at 100M scale (domain shifts vs. synthetic
   fact sets vs. both).
4. Exact flip-persistence metric definition (window length, per-layer vs.
   global).
