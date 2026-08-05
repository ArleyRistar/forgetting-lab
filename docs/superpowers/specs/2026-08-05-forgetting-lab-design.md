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
- **H3 (tertiary, unlocked only by phase-2 escalation):** does the route to
  ternary — born-ternary (from-scratch QAT) vs. converted-from-float — change
  forgetting behaviour? No published work touches this.

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
  fine-tuning only. Local ternary ceiling ≈ **350–560M**. Full QAT training of
  a 360M model ≈ 3–4 GB with 8-bit Adam — Route B fits locally with headroom.
- Pretraining a ~100–135M model from scratch on a few billion tokens: feasible
  (order 50–100 unattended GPU-h per model) — **Route A, only if escalated**.
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

## 6. Phase 1 — build the instrument (weeks 2–8)

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

**1c. Ternarize-and-own (Route B).** Convert a public float model
(SmolLM2-360M) to ternary via continued QAT — BitLinear injection plus the
quantization-warmup recipe from the
[HF Llama3→1.58bit work](https://huggingface.co/blog/1_58_llm_extreme_quantization).
At the moment of conversion the float weights *are* the initial latent weights,
so we own the full latent trajectory without pretraining anything — necessary
because **no public ternary checkpoint releases usable QAT latent weights**
(verified 2026-08-05; see §11). The float twin is the same SmolLM2-360M
continued-trained in bf16 **on the same tokens** (data-matched — otherwise the
conversion corpus confounds the comparison). Measure and report the conversion
quality gap. Shakedown the recipe at 135M first. Save full latent checkpoints
throughout. Budget: ~30–80 unattended GPU-h for the pair. From-scratch
pretraining (Route A) is explicitly deferred to phase-2 escalation.

**1d. Instrumentation.** Per layer, per logging interval: ternary flip
fraction, flip persistence, latent distance-to-threshold histograms; for the
float twin: L2/cosine parameter distance. Note: ternarization uses a per-step
absmean scale, so thresholds move — a weight can flip without moving.
Instrument accordingly (flips are defined by effective-value change, not
latent-value change).

**Eval design:** the converted 360M pair retains its parent's capabilities, so
benchmark subsets become usable where a from-scratch 100M model would sit at
chance. Likelihood-based probes — held-out-fact NLL, per-token loss on task
data — remain primary (they stay sensitive where accuracies floor or
saturate), with chance-adjusted accuracy metrics (framework:
[2510.17776](https://arxiv.org/abs/2510.17776)) wherever accuracy is used.

Deliverables: rig re-runnable from a commit hash; calibration note (blog #1);
the converted ternary/float pair with full latent history from conversion
onward.

## 7. Phase 2 — the experiment (months ~3–6)

Sequential fine-tuning of the converted 360M pair (domain-shift LM corpora +
synthetic fact-recall probes; final design via design cards). ≥3 seeds on all
result-bearing runs; conclusions from paired comparisons only.

- **Primary analysis (H1):** predictive power of flip-fraction/persistence for
  per-task forgetting in the ternary model vs. parameter-distance predictors in
  the float twin — per layer and global.
- **Secondary (H2):** forgetting-curve shape comparison at matched capability —
  magnitude, burstiness, learning-rate sensitivity.
- **Escalation ladder (each step needs its own design card):**
  1. Route A — from-scratch matched pair at ~100M, only if the Route B pilot
     shows signal; unlocks H3 (provenance comparison, converted vs. born-ternary).
  2. Rented arm — BitNet-b1.58-2B4T from its **true bf16 master weights** (the
     only public ternary checkpoint with real latents; full QAT fine-tune fits
     a rented 24 GB card): the same measurements on a production-grade
     born-ternary model.
  3. Behavioural-only confirmatory arm on Spectra TriLMs 99M–560M and/or
     Falcon-E `prequantized`, with the centroid-initialisation caveat
     documented.

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

- **QAT recipe fiddliness** — ternary conversion is documented (HF recipe,
  onebitllms) but hyperparameter-sensitive, and warmup-quantization behaves
  differently at small scale. Mitigation: shakedown at 135M; budget one restart.
- **Conversion-gap confound** — the ternary twin starts weaker than its float
  twin. Mitigation: data-matched twins (§6 1c), report the gap, and measure
  forgetting relative to each model's own post-conversion baseline, never
  cross-model absolutes.
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

## 11. Direction review — RESOLVED 2026-08-05: Route C (raised 2026-08-05)

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

**Decision (Arley, 2026-08-05): Route C.** Route B first — ternarize
SmolLM2-360M via continued QAT (§6 1c): cheap, real latents from conversion
step zero, evals above floor, and it directly studies the conversion route the
objection considers realistic. Route A (from-scratch matched pair) deferred to
phase-2 escalation, where provenance becomes hypothesis H3. (Considered
alternatives: A-first — cleanest control but ~150 GPU-h before any signal and
floor-level evals at 100M; B-only — loses the provenance question.)

**Falcon-E verification (2026-08-05):** its bfloat16 revision is **not** the
QAT master weights. The
[Falcon-E blog](https://falcon-lm.github.io/blog/falcon-edge/) describes it as
a scale-injection *approximation* reconstructed from the ternary weights
("injecting the weight scale after quantizing the weights should lead to a
good enough 'approximation' of the non-BitNet version"), and
[onebitllms](https://github.com/tiiuae/onebitllms) fine-tunes from the
`prequantized` revision (LoRA explicitly unsupported). So the "no public
latents" premise stands — with one exception:
[BitNet-b1.58-2B4T-bf16](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)
ships genuine master weights, too large for 8 GB but viable on a rented 24 GB
card → recorded as escalation step 2 in §7.

## 12. Open questions (resolved via design cards, not now)

1. Conversion corpus and token budget for the 360M pair (candidate:
   FineWeb-Edu sample; set after the 135M shakedown).
2. QAT codebase: onebitllms vs. the HF Llama3→1.58bit recipe vs. a nanotron
   BitNet recipe — decide in phase 1a after reading all three.
3. Task-sequence design at 360M scale (domain shifts vs. synthetic fact sets
   vs. both).
4. Exact flip-persistence metric definition (window length, per-layer vs.
   global).
