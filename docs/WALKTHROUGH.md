# How the smoke-test training run actually works

Phase-0 deliverable: enough understanding of the stack to direct experiments
confidently. Read alongside `src/flab/train.py` and `src/flab/data.py`. Every
choice here reappears in phase 1, so the vocabulary is the point.

## 1. What the model is, before we touch it

`HuggingFaceTB/SmolLM2-360M` is a 360-million-parameter decoder-only
transformer trained to predict the next token. Not chat-tuned — it continues
text. Loaded in **bfloat16**: 16 bits per weight, so ~0.7 GB of VRAM for
weights. bf16 has the same exponent range as fp32 (so it doesn't overflow
easily) but fewer mantissa bits, which is the standard trade for training.

## 2. What the data module does (`data.py`)

`load_smoltalk()` pulls conversations and flattens each into one plain string:

```
<|user|>
What is the capital of France?
<|assistant|>
Paris.
<|end|>
```

Those angle-bracket tags are **not** special tokens the model knows — they're
ordinary text that tokenizes into several tokens each. The model learns "after
`<|assistant|>` comes an answer" purely from seeing the pattern repeatedly.
That's all instruction-tuning is at this level: teaching a text-continuer that
one particular textual shape means "now answer".

We keep 4000 examples for training and 200 held out for eval. Held-out data is
the only honest signal — training loss always falls, including when the model is
memorising rather than learning.

## 3. What LoRA is, and why we use it

Full fine-tuning updates all 360M weights: you store the weights, their
gradients, and two optimizer moments per weight — roughly 4× the model in
memory. LoRA (Low-Rank Adaptation) **freezes the original weights entirely** and
injects a small trainable detour beside each target matrix.

For a weight matrix `W` of shape `d×k`, LoRA learns two thin matrices `A` (r×k)
and `B` (d×r) with `r=16`, and computes `W·x + (alpha/r)·B·A·x`. Because `r` is
tiny next to `d` and `k`, `A` and `B` together hold well under 1% of the
parameters. Only they receive gradients.

Three consequences that matter for the research:

1. **Memory**: optimizer state is proportional to trainable parameters, so it
   nearly vanishes. This is why a 360M model trains comfortably in 8 GB.
2. **The base model is untouched.** The adapter is a separate ~10 MB file. You
   can attach, detach, or swap it — which is exactly why the continual-learning
   literature reaches for LoRA when studying sequential tasks.
3. **It constrains what can be learned.** Updates are confined to a rank-16
   subspace. That's a limitation for capability, but for us it's *also a
   variable* — how much a model can forget is bounded by how much it can change.

`target_modules` lists which matrices get adapters: the four attention
projections (`q,k,v,o`) and the three MLP ones (`gate,up,down`) — i.e. every
linear layer in the block, the standard aggressive choice.
`lora_alpha=32` with `r=16` gives a scaling factor of 2.

## 4. What one training step does

Read `SFTConfig` in `train.py` as the description of a single step:

- `per_device_train_batch_size=4` — four sequences go through the GPU at once.
- `gradient_accumulation_steps=4` — we do that four times, summing gradients,
  before updating. So the **effective batch is 16 sequences**, while only 4 are
  ever in memory. This is the standard way to buy a large batch on a small card.
- `max_length=1024` — sequences are truncated/padded to 1024 tokens. Attention
  cost grows quadratically with length, so this is a major VRAM lever.
- `learning_rate=2e-4` with `lr_scheduler_type="cosine"` and `warmup_steps=20` —
  start near zero, ramp up over 20 steps (early full-size updates on a fresh
  adapter destabilise training), then decay smoothly to zero. 2e-4 is high for
  full fine-tuning and normal for LoRA, because the adapter starts from zero and
  has fewer parameters to move.
- `gradient_checkpointing=True` — **the memory-for-compute trade.** Normally
  every intermediate activation from the forward pass is kept so the backward
  pass can reuse it. With checkpointing, most are thrown away and recomputed
  during backward. Roughly 30% slower, and it cuts activation memory sharply.
  On 8 GB this is usually what makes a run fit at all.
- `seed=0` — fixes initialisation and data shuffling. Non-negotiable for us:
  comparing two runs is only meaningful when the only difference is what we
  changed.

The loop itself: forward pass computes next-token predictions and a
cross-entropy loss against the true next tokens; backward pass computes the
gradient of that loss with respect to the LoRA parameters only; the optimizer
nudges them downhill. Repeat 400 times.

## 5. Checkpoints and why resume matters

`save_steps=100` writes the full trainer state — adapter weights, optimizer
moments, scheduler position, step count — every 100 steps.
`trainer.train(resume_from_checkpoint=True)` reads the newest one and continues
where it stopped, rather than restarting.

This isn't housekeeping, it's a research requirement. Phase 1 runs jobs for many
hours unattended; a crash at 3 a.m. without resume costs a day of calendar time
for a few seconds of compute. We verified it deliberately by killing a run
mid-flight and restarting it.

`eval_strategy="steps"` with `eval_steps=100` measures held-out loss at the same
cadence, so we can see the train/eval gap open up as it overfits.

## 6. How we measure whether anything happened

`lm-evaluation-harness` runs standardised benchmarks. We evaluate the same base
model twice — once bare, once with the adapter attached — which makes the
comparison paired: identical weights underneath, one difference.

The tasks split by capability type, which is the distinction the whole research
programme rests on:

- **arc_easy, hellaswag** — knowledge and commonsense. Scored by comparing the
  likelihood the model assigns to each candidate answer. No text generation, so
  fast.
- **ifeval** — instruction-following. The model must *generate* text obeying
  verifiable constraints. Slow, and near zero for a base model, so it's where
  fine-tuning should show the largest jump.

Expected shape: IFEval rises sharply, arc_easy/hellaswag barely move. If
general benchmarks *drop*, you have just observed catastrophic forgetting in
miniature — which is the entire subject of this lab.

## 7. What this teaches us for phase 1

The same skeleton becomes the sequential-fine-tuning harness: instead of one
task, we run task A → task B → task C on the same model, evaluating after each
stage. Everything above stays; what gets added is the loop over tasks, the
per-stage evaluation matrix, and — for the ternary work — instrumentation that
watches which weights *flip* rather than how far they move.
