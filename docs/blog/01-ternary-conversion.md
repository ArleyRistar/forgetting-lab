# Perplexity recovery is not capability recovery: converting a 360M model to ternary, with a control

*Draft — Claude wrote this; Arley edits and decides what is claimed. Every number
traces to `docs/LAB-NOTES.md`. Anything the notes mark retracted or inconclusive
is written that way here.*

---

## TL;DR

We converted SmolLM2-360M to ternary (1.58-bit) weights with continued
quantisation-aware training, and trained a **float twin on exactly the same
tokens** as a control. Matched-precision suites exist
([Spectra](https://arxiv.org/abs/2407.12327) trains FloatLM and TriLM on the same
data); what we could not find is a *conversion* study that runs the float control
at a matched token budget. Then we spent
several days trying to measure whether ternary models forget differently, and
mostly found out that we were measuring the wrong things.

What we can report:

1. **A bf16 trap that silently kills conversion.** With bf16 latent weights,
   **85% of Adam updates round to zero** at lr 1e-4 (96% at 2e-5). Three
   conversion runs failed before we found it. Changing only the dtype fixed it.
2. **Perplexity recovery is not capability recovery.** Our converted model
   reaches a sane-looking loss and can do **nothing**: zero discrimination on
   every pretrained task we tested, while the float twin keeps 5.5–9.4 SE.
3. **But it can still learn.** The same checkpoint memorises 50 arbitrary
   key→value associations to perfect held-out accuracy. Recall gone, plasticity
   intact.
4. **Weight-state flips are a rescaled L2 distance.** `flips / L2` is
   0.0002 ± 15% across both arms, every cadence, and a 20× range of L2. Our
   primary hypothesis — that flips predict forgetting better than parameter
   distance — is **refuted**.
5. **Ternary forward passes are not batch-invariant.** Scoring identical prompts
   at batch 1 versus batch 8 moves log-probs by a median of **0.34 nats** in the
   ternary model and **1.1e-5** in the float twin. Activation quantisation is the
   cause. If you evaluate a low-bit model on likelihood, this affects you.

And one claim we published and then **retracted**, which is written up here
because the retraction is more useful than the claim would have been.

---

## Why a float twin

Go and read the discussion tabs on any ternary model release and you will find
the same argument. From `microsoft/bitnet-b1.58-2B-4T` #15:

> *"I don't think there is any benefit in training efficiency since it still use
> full precision in training stage"* — to which another commenter replies:
> *"otherwise why even bother training a new model... [the authors] could have
> just applied post-training ternary quantisation [to an existing model like
> Phi-4]."*

The question underneath it is: **how much of what you lose is ternarisation, and
how much is just the extra training you did?** You cannot answer that from a
conversion run alone. So we ran two:

| | ternary twin | float twin |
| --- | --- | --- |
| base | SmolLM2-360M | SmolLM2-360M |
| corpus | FineWeb-edu, 65.5M tokens | **the same 65.5M tokens, same order** |
| steps | 4000 | 4000 |
| lr / schedule | 1e-4, cosine | 1e-4, cosine |
| optimiser | 8-bit Adam, β₂=0.95, wd 0 | identical |
| difference | BitLinear at λ=1 | nothing |

Everything matched except the thing under test. Where the two arms differ, they
differ in ways the experiment does not depend on: **micro-batch** (2×8 vs 4×4 —
the float arm OOMs at the ternary arm's, for a reason we get to below), allocator
configuration, and therefore step time. Tokens, token order, tokens-per-step,
total steps, LR, schedule, optimiser and seed all match.

**One deliberate choice worth flagging:** both arms use the same learning rate,
following [Nielsen et al.](https://arxiv.org/abs/2502.11895), so the pair differs
in one variable rather than two. Microsoft's ternary recipes use roughly 6× the
float LR at the same model size, so 1e-4 probably leaves our ternary arm below its
own optimum. That is one reason we make no absolute-quality claim about it.

One caveat before any number: **65.5M tokens is roughly 150× less than the
reference recipe** ([HF's Llama3→1.58bit
work](https://huggingface.co/blog/1_58_llm_extreme_quantization) used ~10B). This
is a single 8GB laptop GPU. Read everything here as "what happens at a hobby
budget", not "what ternary conversion can do".

---

## Finding 1: bf16 latent weights silently destroy the training signal

Our first three conversion attempts failed the same way — loss climbing to
~11.75, above the ~10.8 of a uniform guess. We blamed the missing per-layer norm,
added it, and got 10.59. Still broken.

The actual cause: **the latent weights were bf16**.

In quantisation-aware training the ternary weights are recomputed every forward
pass from a full-precision *latent* weight, and gradients flow to the latent via a
straight-through estimator. bf16 has 8 significant bits. We measured what that
does to an Adam step on this checkpoint:

| latent dtype | lr | Adam updates that round to **zero** |
| --- | ---: | ---: |
| bf16 | 1e-4 | **85.4%** |
| bf16 | 2e-5 | **96.3%** |
| fp32 | 1e-4 | — (works) |

The model was freezing most of its latent weights every step — up to 24 in 25 at
lr 2e-5. And it explains
the thing that had confused us most: **lowering the learning rate made it worse**,
because smaller updates round away more often.

Changing the dtype — with the learning rate held at 1e-4 — fixed conversion. The
135M shakedown went from "collapsed to chance and stayed there" to a working
model. (That run also reverted an earlier experiment of ours with the layer norm,
so dtype is the change that mattered, not the only change we made.)

Every reference implementation trains an fp32 master (nanotron defaults
`accumulate_grad_in_fp32: true`), so this is not a new technique — but nothing we
found states the failure mode, and it is silent. The loss curve looks like a model
that is training badly, not like a model whose gradients are being discarded.

**Scope it honestly:** this is conditional on *bf16 + small batch + lr 1e-4*. At
Microsoft's 1M-token batch and lr 1.5e-3 the per-step update is orders of
magnitude larger and bf16 would not round it away. The claim is not "bf16 latents
are broken" — it is "at hobby batch sizes they are, and the failure is invisible".

---

## Finding 2: perplexity recovery is not capability recovery

After 4000 steps the converted model looks like it worked. Held-out loss on
FineWeb-edu:

| model | held-out loss | perplexity |
| --- | ---: | ---: |
| SmolLM2-360M (base) | 2.5257 | 12.50 |
| float twin | 2.5373 | 12.65 |
| **ternary twin** | **5.1288** | **168.82** |

The float twin lands **+0.0116 nats** from base — given the same 65.5M tokens, it
went essentially nowhere. That is the control doing its job: the 2.59-nat gap
belongs to **ternarisation**, not to the extra training, the corpus, or the
learning rate.

Out of distribution the gap is worse — 3.63 nats on WikiText-103 — so the
converted model has partly re-fit to its conversion corpus rather than retained
general ability.

Then we asked whether it could actually *do* anything. For each task we compared
the model's answer NLL against a **shuffled-answer control** (a derangement, so no
item keeps its own answer): a model with task knowledge should score the true
pairing better. We ran the base model too, as a positive control for the test
itself.

| task | base | ternary twin | float twin |
| --- | ---: | ---: | ---: |
| ScienceQA | **6.1 SE** | 0.8 SE | **5.5 SE** |
| Py150 | **9.4 SE** | 0.1 SE | **9.2 SE** |
| NumGLUE-cm† | 1.5 SE | 0.2 SE | 1.1 SE |
| FOMC | 0.1 SE | −0.8 SE | 0.1 SE |

**The ternary twin discriminates on nothing.** The base model discriminates
strongly on two of four, so this is a statement about the model rather than about
our instrument. And the float twin — same tokens, same steps — keeps essentially
all of it.

Py150 makes the size of it plain: answer NLL is 2.31 for base, 2.45 for the float
twin, **8.29** for the ternary twin, and the discrimination signal collapses from
+1.69 nats to **+0.017**.

So: a loss curve that converges, a perplexity that is merely bad, and **zero
measurable capability**. If you are converting a small model on a small budget,
perplexity will not tell you this has happened.

† NumGLUE-cm has only 41 scorable held-out items, so it is **underpowered, not
evidence of absence** — base's effect there would reach ~3.3 SE at n=200.

(FOMC fails for *every* arm including base — it is a bad probe at this scale, not
evidence about any model. We report it because dropping it silently would be
cherry-picking.)

---

## Finding 3: recall is gone, plasticity is not

Given that, we assumed our forgetting experiment was dead — you cannot measure
forgetting in a model with nothing to forget.

That was wrong, and the error is instructive. The capability gate asks *can it
recall?*. A forgetting experiment on **taught** tasks asks *can it learn?*. Nobody
had asked the second question.

We fine-tuned each twin for 300 steps on 50 nonsense key→value associations
(single-letter answers, so chance is exactly log 8 = 2.0794 nats):

| arm | held-out NLL before | after | token accuracy | shuffled-control margin |
| --- | ---: | ---: | ---: | ---: |
| ternary | 7.21 | **0.000057** | **1.000** | +11.70 (17.7 SE) |
| float | 6.11 | **0.000036** | **1.000** | +13.93 (17.5 SE) |

**Perfect recall of all 50 associations, in both arms.**

So the converted model has **no retained capability and full plasticity**. The
same checkpoint that cannot discriminate on a single pretrained task memorises
fifty arbitrary facts to perfect held-out accuracy. Conversion at this budget
destroyed what the model *knew*, not its ability to *learn*.

We have not seen this dissociation reported, and it has a practical reading: if
you are converting a small model in order to fine-tune it on your own task
anyway, the catastrophic-looking capability loss may matter less than the
perplexity suggests.

---

## Finding 4: weight-state flips are a rescaled L2 distance

This was the project's primary hypothesis, and it is refuted.

Ternary weights are `{-s, 0, +s}` where `s` is the mean absolute weight,
**recomputed every forward pass**. So the decision boundary moves under the
weights: a weight's effective state can flip without the weight moving much. That
seemed like it might make forgetting behave differently from float models, and
"fraction of weights whose state flipped" seemed like it might predict forgetting
better than plain parameter distance.

We built the instrument, partitioned every flip into four disjoint causes
(weight-driven, threshold-driven, either, both), and measured it across
conversion, a no-distribution-shift null arm, and 72 sequential fine-tuning runs.

**A scoping note that matters for reading the numbers:** 224 BitLinear layers
cover **86.9%** of the parameters. The tied embedding/`lm_head` — 13.1%, and it
*is* the output head — stays float, as in every reference recipe. Both the flip
and the L2 statistics below cover the ternarised core only, so do not compare them
against a whole-model parameter distance.

**Two results, both negative, both clean.**

The threshold-motion confound is negligible. Scale-driven flips are **0% per
step**, rising only to 0.0084% of flips at 1000-step intervals. The worry that
motivated the whole decomposition does not materialise — though it was worth
building, because we could not have known that in advance.

And the metric carries no information beyond L2:

| regime | flips / L2 |
| --- | ---: |
| conversion, 1000-step intervals | 0.000186–0.000208 |
| null arm, 25-step intervals | 0.000208–0.000249 |
| task shift, 72 runs | 0.000198–0.000222 |

Across a 20× range of L2, both arms, every cadence, **the ratio does not move**.
Flip fraction is a fixed rescaling of parameter distance, so it cannot predict
anything L2 does not. Directly: in the ternary arm, flips and L2 correlate with
forgetting at +0.5952 and +0.6006 — indistinguishable, flips marginally worse.

If you were hoping weight-state flips are a cheap, scale-free forgetting
predictor: on this evidence — a mid-conversion twin at lr 1e-4, on synthetic
recall tasks — they are L2 in different units.

---

## Finding 5: ternary forward passes are not batch-invariant

We found this by accident, chasing a number that looked wrong, and it is probably
the most immediately useful thing here.

Score the same 64 prompts through the same ternary checkpoint at batch 1 and at
batch 8. The log-probabilities differ by a **median of 0.34 nats**. Do it with the
float twin: **1.1e-5**. About 30,000×.

| batch | ternary median drift | float median drift |
| ---: | ---: | ---: |
| 2 | **0.364** | 1.14e-5 |
| 8 | 0.344 | 1.14e-5 |
| 32 | 0.357 | 1.34e-5 |

Three things we checked, because the obvious explanations are wrong:

- **It is a step change, not a gradient.** Batch 2 is already as bad as batch 32.
  Using smaller batches does not help; only batch 1 does.
- **It is not padding.** With equal-length inputs and no padding at all, the drift
  is unchanged (0.342). Bucketing by length will not fix it, and neither will
  attention masks.
- **It is activation quantisation.** Disable per-token absmax activation
  quantisation and drift drops to float levels (1.5e-5). Disable *weight*
  quantisation instead and 0.062 remains.

### It is not the batch-coupling bug you are thinking of

There is a well-known way for quantised inference to become batch-dependent: a
quantisation scale computed *across* the batch, e.g. `scale = x.abs().max()` over
the whole tensor, so one row's outlier changes another row's quantisation. That
is a filed bug ([bitts#2](https://github.com/jaweed3/bitts/issues/2)) and now a
security paper — [*Quantamination: Dynamic Quantization Leaks Your Data Across
the Batch*](https://arxiv.org/abs/2604.26505) shows several mainstream frameworks
ship configurations that leak across the batch boundary.

**That is not what this is.** Our activation scale is per-token:

```python
scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
```

`dim=-1` means each row's scale sees only that row. Measured, not just argued: a
row quantised alone is **bit-identical** (max difference exactly 0.0) to the same
row batched alongside rows 120× larger. The drift survives a scale with no
cross-batch path — which is what makes it a different phenomenon, and a harder
one to design away.

### The mechanism

cuBLAS returns slightly different results for different batch shapes — ~1e-7 per
matmul, accumulating to the ~1e-5 the float twin shows end to end. Then `round()` in activation quantisation
snaps to int8 levels, so a perturbation near a rounding boundary flips a level by
1/127 ≈ 0.008 — an amplification of roughly 10⁵ — and 224 quantised layers
compound it.

The amplification principle is not new: [Defensive
Quantization](https://arxiv.org/abs/1904.08444) (Lin, Gan & Han, ICLR 2019) named
the "error amplification effect", where quantisation enlarges perturbations layer
by layer and worse the fewer the bits. They applied it to adversarial noise; we
are applying it to the float noise that batch shape already produces.

### It does not change what the model says

**Zero argmax flips out of 64 for both shipped models** — every batch size, and
with padding removed. Generation and greedy accuracy are unaffected.

(The weight-quant-off *diagnostic* ablation did flip 18 of 64. That is not a model
anyone ships — it is a probe we ran to isolate the cause — but it is in the output
JSON, and we would rather say so than have you find it.)

So the consequence is narrow but real: **anything likelihood-based on a low-bit
model is batch-dependent at the ~0.3-nat level.** Perplexity comparisons, NLL
probes, KL measurements. A few tenths of a nat between two ternary checkpoints may
be your batch size rather than your checkpoints. Score at batch 1, or report the
batch size as part of the measurement.

### What is and is not new here

Batch-dependent inference in *float* LLMs is well documented — [Thinking
Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and [arXiv 2506.09501](https://arxiv.org/abs/2506.09501), the latter reporting up
to 9% accuracy swings from batch size, GPU count and GPU version. Neither touches
quantisation. Batch-invariance bugs in quantised *kernels* are also known and
fixed at the engineering level — see vLLM
[#36488](https://github.com/vllm-project/vllm/pull/36488), where an mxfp4 MoE
Triton kernel picked its block size from tokens-per-expert and 20 of 30 prompts
changed — but that is accumulation order, not rounding amplification, and nobody
there quantifies quantised-versus-float drift.

What we believe is new is the measurement and its attribution: the ~30,000×
ternary-versus-float ratio, activation quantisation isolated as the cause *under
a batch-independent per-token scale*, the step change at batch 2, the padding
rule-out, and the consequence for likelihood-based low-bit evaluation.

**One scope limit that matters.** This is *simulated* quantisation — fake-quant
executed in floating-point kernels, which is what QAT and most research code do.
A true integer pipeline may behave differently: [arXiv
2607.23227](https://arxiv.org/abs/2607.23227) argues INT8 QDQ on ARM is
dispatch-invariant, with every intermediate accumulator byte-identical, precisely
because discrete inputs remove the float noise our mechanism needs. So read this
as a claim about simulated-quantisation research pipelines, not about deployed
integer inference.

---

## What we got wrong

We published "the ternary model forgets less" and retracted it. The retraction is
worth more than the claim was.

We had trained both twins on task A, then on a conflicting task B that overwrites
A's answers, and measured how much A's NLL rose. The ternary twin's rose by ~6
nats less. Both arms *appeared* to have mastered B identically. It looked clean.

**Then someone checked the accuracy column.** Task-A token accuracy was **0.00 in
both arms, every seed**. Both models had forgotten A *completely*.

The "identical B mastery" was wrong too: the NLLs differed 7× (0.000532 vs
0.000075), and because A's NLL is bounded below by −log(1−p(B)), that gap alone
mechanically forces about 2 of the 6 nats. The rest lived in the log-probability
of a token neither model would ever emit — at 7–14 nats above chance, comparing
p≈1e-4 against p≈1e-7. Softmax tail geometry, not memory.

And where retention *was* measurable — a task pair that does not overwrite — the
direction reversed: the float twin retained 0.60 accuracy against the ternary
twin's 0.40.

The accuracy was in every one of the 72 output files the entire time. We analysed
the NLL and never opened it.

This is the same shape as a bug we had documented two days earlier, where a
50-point "accuracy collapse" turned out to be a turn-terminator token. Both times
the honest number was sitting in the output. So we added a guard that refuses to
report a capability claim when accuracy is at floor in every arm, or when NLL sits
more than 2 nats above a task's chance level. It is ~100 lines, and it would have
caught this retraction. The turn-terminator bug needed a different check — content
accuracy, excluding the answer's final token — which also now exists.

**If you take one methodological thing from this post:** a likelihood difference
between two models is not a capability difference, and the further you are from
chance the less it means. Check the behavioural number before you believe the
probabilistic one.

---

## What we did not establish

- **Whether a sub-behavioural trace survives overwriting.** We measured one, using
  fixed effects over letters and keys against a matched placebo. It holds for the
  **float** twin (t=20 on a rank-based, calibration-free statistic) and **not** for
  the ternary twin (t=1.8, confidence interval spanning zero). An earlier
  log-units version showed a large ternary effect; it did not survive a scale-free
  test, because the two arms' log-probability scales differ by ~2.5 nats on
  never-taught letters. Cross-arm magnitude comparisons in log units are exactly
  the error that produced the retraction above.
- **Anything about ternary models in general.** One base model, one size, 65.5M
  tokens, one 8GB GPU.
- **That conversion "fails" at small scale.** The reference recipe uses ~150× more
  tokens, and published work shows converting a *fully*-trained model is the worst
  point on the transition curve ([Nielsen et
  al.](https://aclanthology.org/2025.findings-acl.694/)).
- **The KL replication** from our earlier calibration work sits 5.4× above the
  figure in [arXiv 2606.27634](https://arxiv.org/abs/2606.27634), unexplained. The accuracy replicates; the KL does not.

---

## Reproducing

Everything is in [the repo] — conversion, the twin, the flip instrument, the
capability gate, and the lab notes with every number, including the ones we
retracted.

```bash
uv run python -m flab.convert --model HuggingFaceTB/SmolLM2-360M --mode ternary \
  --max-steps 4000 --lr 1e-4 --optim adamw_bnb_8bit
uv run python -m flab.convert --model HuggingFaceTB/SmolLM2-360M --mode float \
  --max-steps 4000 --lr 1e-4 --optim adamw_bnb_8bit \
  --batch-size 2 --grad-accum 8 --expect-tokens-per-step 16384
```

Total compute for everything described here: roughly 35 GPU-hours on one RTX 3070
Ti Laptop, over about a week.

## Related work worth reading

- [Laborieux et al., *Synaptic Metaplasticity in Binarized Neural
  Networks*](https://arxiv.org/abs/2003.03533) — our mechanism, one bit lower and
  five years earlier: hidden real-valued weights as metaplastic variables, with
  weights far from the threshold made to resist flipping.
- [Helwegen et al., *Latent Weights Do Not
  Exist*](https://arxiv.org/abs/1906.02107) — latent weights as inertia rather
  than weights; the licence for talking about flips at all.
- [Spectra / TriLM](https://arxiv.org/abs/2407.12327) — already owns the
  matched-precision-suite idea. Our twin is not a novel control design; applying
  it to forgetting is the new part.
- [*When Less is More: 8-bit Quantization Improves Continual
  Learning*](https://arxiv.org/abs/2512.18934) — points the opposite way to our
  intuition, and we could not confirm or refute it at this scale.
- Nielsen, Schneider-Kamp & Galke, [*Continual Quantization-Aware Pre-Training:
  When to transition from 16-bit to 1.58-bit pre-training for BitNet language
  models?*](https://arxiv.org/abs/2502.11895) — the closest published recipe to
  ours, with full hyperparameters. Findings of ACL 2025.
