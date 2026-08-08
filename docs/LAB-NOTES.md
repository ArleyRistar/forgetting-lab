# forgetting-lab — lab notes

Running notebook for the lab box (`gs66-lab`, MSI GS66 12UGS, RTX 3070 Ti
Laptop 8 GB). Newest entries at the bottom.

## 2026-08-07 — bring-up + thermal characterisation (phase 0, tasks 1–2)

### Environment

- Fedora 44, kernel 7.1.6-201.fc44, headless (`multi-user.target`, gdm masked).
- NVIDIA 610.57.04, CUDA 13.3 runtime. torch **2.13.0+cu130**.
- **Usable VRAM: 7.66 GiB total, 7.50 GiB free** — plan against 7.5, not 8.
- Verified end-to-end: a real bf16 8192² matmul executes on the GPU
  (`nvidia-smi` alone does not prove the CUDA userspace works).

### Thermals — 10-min bf16 matmul burn (`scripts/burn.py`), uncapped

| metric | value |
| --- | --- |
| peak GPU temp | **87 °C** (plateau, not a climb) |
| peak CPU package | **83 °C** (with the CPU only feeding the GPU) |
| sustained power | 80 W, self-trimming to 72 W at the plateau |
| SM clock under load | 1125–1400 MHz (max 1635) |
| idle → load ramp | firmware raises its own ceiling 45 W → 80 W |
| cooldown | 79 → 76 °C in 12 s after load stopped |

Findings:

1. **`nvidia-smi -pl` is not supported on this vBIOS** ("not supported in
   current scope"). The firmware manages the 45–105 W envelope itself via
   `nvidia-powerd` (Dynamic Boost). The clock cap (`-lgc`) *is* supported and is
   therefore the only throttle available.
2. The 87 °C plateau is **soft-regulated, not distress**: `SW Thermal Slowdown`
   goes Active and trims power/clocks, while `HW Thermal Slowdown` logged
   **0 µs — never engaged**. GPU thresholds are 95 °C slowdown / 98 °C shutdown
   / 105 °C max operating, so there is ~8 °C of margin to the first hardware
   intervention.
3. Cost of running at the wall: clocks fall ~1400 → ~1125 MHz, i.e. roughly
   20% of throughput surrendered to thermal slowdown, **and step times become
   variable** — which is the real problem, because it makes GPU-hour estimates
   on design cards unreliable.
4. This burn is a deliberate worst case (continuous matmul, no idle gaps). Real
   LoRA fine-tuning will run cooler; measured separately in task 4.

Action taken: `nvidia-powercap.service` now applies persistence mode **plus a
1200 MHz SM clock cap** (inside the range the thermal governor was already
choosing, so little throughput is lost while boost overshoot is removed).
Reset with `sudo nvidia-smi -rgc`; relax the cap if task 4 shows real training
runs cool.

Maintenance note: chassis is ~4 years old and the near-idle CPU sitting at
80 °C is consistent with heat soak through the shared heatpipe and/or aged
thermal paste. Not urgent (nothing throttled in hardware), but dust removal
from fans/fins is cheap and a repaste would likely buy back some of the lost
20%. Flagged to Arley 2026-08-07, no action taken.

### Instrumentation quirks worth remembering

- `nvidia-smi --query-gpu=power.draw` returns **garbage on its first sample**
  (~751 W) then settles (~19 W idle). `scripts/burn.py` discards sample one.
- `power.limit` reads `[N/A]` on this card; use
  `nvidia-smi -q -d POWER | grep "Current Power Limit"` instead.
- At 0% utilisation clocks jump to ~1620 MHz — high idle clocks are not a sign
  of load, so read utilisation alongside clocks.

## 2026-08-07 — smoke run (phase 0, task 4)

400-step LoRA SFT of SmolLM2-360M on `smol-smoltalk[:4000]`, seed 0, r=16,
effective batch 16 (4 × 4 accumulation), max_length 1024, bf16 + gradient
checkpointing, 1200 MHz clock cap active.

| metric | start | end |
| --- | --- | --- |
| train loss | 1.456 (@20) | **1.255** |
| eval loss (held-out 200) | 1.362 (@20) | **1.187** |
| eval token accuracy | 0.6612 | **0.6903** |

- **Wall clock: 50 min 16 s** for 400 steps (3016 s), average 7.54 s/step.
- **Peak temps: 87 °C GPU, 95 °C CPU package.**
- **VRAM: 1.86 GiB of 7.66 GiB.** Enormous headroom — the 8 GB card is nowhere
  near the constraint at this scale. Revisit the spec's §4 envelope: bigger
  batches, longer sequences, or a larger model are all affordable.
  **Correction (2026-08-07, see the re-derivation section below): the claim
  originally made here — that ternary QAT "has far more room than the
  350–560M ceiling previously estimated" — was over-read from a LoRA run.
  A LoRA measurement constrains the activation term but says nothing about
  full-fine-tune optimizer memory, which is what sets the QAT ceiling.**
- Adapter size: **34.8 MB** vs ~720 MB for the bf16 base model.

### Thermal derate is the real constraint, not memory

Step time degrades as the chassis heat-soaks — this is the single most important
operational fact for planning experiments:

| elapsed | step time |
| --- | --- |
| cold (steps 1–30) | 4.85 s |
| ~10 min | 7.05 s |
| ~20 min | 7.60 s |
| ~25 min | 9.43 s |

That is up to **1.9× slower than cold**, averaging 7.54 s over the run. Steady
state takes 10+ minutes to reach, so **any thermal or throughput reading taken
before ~15 minutes is misleading** (this was misjudged twice during bring-up).

**Use ~1.9× the cold-start estimate when budgeting GPU-hours on design cards.**

Mitigations: no Linux fan control exists (`msi_wmi_platform` exposes RPM
read-only, no PWM; fans observed at 3600–4200 RPM against a ~5500–6000 ceiling),
and the CPU is already 94% idle on the `powersave` governor at 400 MHz — so the
92–95 °C CPU reading is GPU heat soaking through the shared heatpipe, not CPU
work. There is no software fix. Physical remediation only: fan/fin cleaning
(Arley planning), possible repaste, BIOS fan profile / Cooler Boost.

Untested idea: a *lower* clock cap may raise average throughput by preventing
the boost → overheat → hard-throttle oscillation. Worth an A/B (e.g. 1000 vs
1200 MHz over 50 steps each).

## Environment quirks — read before debugging anything

1. **Python is pinned to 3.12** (`.python-version`). Fedora 44's system Python
   is 3.14 with no dev headers, which makes triton fail to JIT-compile its CUDA
   shim (`fatal error: cuda.h` / missing `Python.h`) and kills any training run.
   Do not unpin: the low-bit packages phase 1 needs (`onebitllms`, bitsandbytes)
   have no 3.14 wheels.
2. **transformers 5.x renamed `torch_dtype` → `dtype`.** `SFTConfig(
   model_init_kwargs={"dtype": "bfloat16"})` is correct here; the older name is
   what most tutorials still show.
3. Installed: torch 2.13.0+cu130, transformers 5.14.1, trl 1.9.2, peft 0.20.0.
   TRL 1.9 uses `max_length` and `eval_strategy` (not `max_seq_length` /
   `evaluation_strategy`).
4. **`nvidia-smi -pl` is unsupported on this vBIOS**; only `-lgc` (clock cap)
   works. `power.limit` reads `[N/A]`; use `nvidia-smi -q -d POWER`.
5. `power.draw`'s **first sample is garbage** (~751 W).
6. HF token is installed (user `arleyristar`); creds in
   `~/secrets/huggingface-account-20260807.txt` on the Zenbook.
7. Long runs: `ssh lab` → `tmux new -As <name>`. The box cannot suspend (sleep
   targets masked) and boots headless to a console.

## 2026-08-07 — evaluation + phase-0 close (tasks 5–6)

- **Plan gap fixed:** IFEval needs optional extras. `lm-eval>=0.4.8` alone dies
  with `ModuleNotFoundError: langdetect`; the dependency must be
  **`lm-eval[ifeval]`** (pulls langdetect + immutabledict). Phase 1 eval
  batteries will need the same.
- **`--batch_size auto` works well** for loglikelihood tasks: ~190 it/s,
  49,669 requests (arc_easy + hellaswag) in ~4 min at 4.3 GiB VRAM. But the
  first progress sample claims a 64-hour ETA — another first-sample artifact,
  same trap as `power.draw`. Ignore the first estimate on any progress bar here.
- **IFEval is the expensive task by far:** generative, 541 prompts at ~12 s
  each ≈ 1 h 50 m per model, GPU only ~52% utilised (sequential decoding, not
  compute-bound). For phase-1 iteration use `--limit` on generative tasks, or
  prefer likelihood-based probes — which the spec already favours for other
  reasons.
- **Clean-clone reproducibility verified:** fresh `git clone` → `uv sync` →
  `pytest` passes (2 tests) on Python 3.12.13 with CUDA reachable.
- Machine config: laptop panel powered off (`bl_power=4`) and
  `consoleblank=60` added to the kernel cmdline, so the console self-blanks on
  every boot.

### Open at phase-0 close

Raised here at close; **current status is tracked in the single list at the
bottom of this file** — do not read this list as live.

1. Before/after eval table — collect once the evals finish.
2. Clock-cap A/B (1000 vs 1200 MHz over ~50 steps each) to test whether a lower
   cap raises *average* throughput by avoiding boost→throttle oscillation.
3. Re-derive the spec §4 VRAM envelope from the measured 1.86 GiB.

## 2026-08-07 — VRAM envelope re-derivation (open item 3)

### What the 1.86 GiB actually decomposes to

The figure came from `nvidia-smi --query-gpu=memory.used`, so it is
**device-level and includes the CUDA context** — it is not
`torch.cuda.max_memory_allocated`. Decomposing the peak (1905 MiB) for the
task-4 run (SmolLM2-360M, LoRA r=16 on 7 projections, batch 4 × seq 1024,
gradient checkpointing, AdamW):

| term | MiB | basis |
| --- | --- | --- |
| CUDA context | 190 | **measured** on the box, not assumed |
| base weights bf16 | 690 | 361.8M × 2 B |
| LoRA weights fp32 | 33 | 8.68M × 4 B |
| LoRA grads fp32 | 33 | |
| Adam m+v fp32 | 66 | 8.68M × 8 B |
| **activations + workspace** | **892** | residual |

Arithmetic check that validates the parameter counts: 8,683,520 LoRA params at
fp32 = 34,734,080 B against an actual `adapter_model.safetensors` of
34,793,120 B (difference is the safetensors header). The adapter is stored
**fp32**, not bf16.

The 892 MiB residual closes plausibly: 384 MiB of bf16 LM-head logits
(4 × 1024 × 49152 × 2 B) + 240 MiB of checkpointed layer boundaries
(32 × 4 × 1024 × 960 × 2 B) + 268 MiB of recompute buffers, attention
workspace and allocator fragmentation.

**Planning fact:** activation memory here is dominated by the **logits**, which
scale with `vocab × batch × seq` and are *independent of model size*. Every
model in the SmolLM2 family shares the same 49k vocab, so this term does not
shrink when you shrink the model — but halving batch or sequence length buys
back ~190 MiB directly.

### The resulting envelope

Weights-and-optimizer budget = 7.5 GiB usable − context − activations
= **6598 MiB**, or **~5.9 GiB with a 10% safety margin** at batch 4 × seq 1024.

Note the context term **cancels**: it is subtracted from the usable total and
also sits inside the measured 1905 MiB, so the budget reduces to
`7680 − 1905 + weights_and_optimizer`. The ceilings below therefore do not
depend on the one quantity that was ever estimated.

| configuration | B/param | ceiling |
| --- | --- | --- |
| LoRA, base resident in bf16 | 2 | ~3.1B |
| Full FT, bf16 + 8-bit Adam | 6 | ~1.04B |
| Full FT, pure bf16 AdamW | 8 | ~780M |
| Full FT, fp32 master + fp32 Adam | 16 | ~390M |
| Ternary QAT, fp32 latent + 8-bit Adam | 8.2 | **~760M** |
| Ternary QAT, fp32 latent + fp32 Adam | 16 | **~390M** |
| Ternary QAT, all-fp32 + materialised copy | 18 | **~350M** |

### The actual finding

**The spec's 350–560M ternary ceiling was not too pessimistic — it was the
all-fp32 end of a range the spec did not know it was quoting.** The ternary
ceiling spans 350M → 760M, a 2.2× range, and *which end you get is a phase-1
recipe decision* (latent-weight precision, optimizer bit-width, whether the
materialised quantized tensor is fused per layer or held for all layers) —
**not a property of the card**. §4 has been rewritten to say this.

Consequence for phase 1c: if the ternarize-and-own recipe wants headroom above
360M, the lever is **8-bit Adam (bitsandbytes)**, which alone moves the ceiling
from ~390M to ~760M. bitsandbytes is installable here (Python is pinned to 3.12
precisely so these wheels exist — see quirk 1).

### Confidence, and what would settle it

The activation term is **measured**. The per-parameter costs are **computed,
not measured** — a LoRA run exercises optimizer state for 8.68M params, 2.4% of
the model, so it cannot validate the full-fine-tune terms that dominate these
ceilings. Treat the table as a design-time budget, not an empirical result.

Cheapest thing that would convert it: a ~20-step full fine-tune of
SmolLM2-360M (no LoRA) reading `torch.cuda.max_memory_allocated`, under both
`adamw_torch` and `adamw_bnb_8bit`. ~5 minutes of GPU, and it directly measures
the 8 vs 16 vs 6 B/param terms. Not run yet — the GPU was busy with the phase-0
evals. **Queued as open item 4.**

### Measured 2026-08-08 (closes open item 4)

`scripts/mem_probe.py`, 20-step full fine-tune, same batch 4 × seq 1024 and
gradient checkpointing as the LoRA run. Measured bytes/param is
`(peak_allocated − 892 MiB activations) / 361.8M`:

| configuration | peak alloc | peak reserved | measured B/param | predicted | ceiling (was) |
| --- | --- | --- | --- | --- | --- |
| bf16 + `adamw_torch` | 3.45 GiB | 3.90 GiB | **7.66** | 8 | 813M (778M) |
| bf16 + `adamw_bnb_8bit` | 3.02 GiB | 3.26 GiB | **6.37** | 6 | 978M (1038M) |
| fp32 master + `adamw_torch` | 6.84 GiB | 7.41 GiB | **17.71** | 16 | 352M (389M) |

**The computed table holds — every configuration within ~10% of prediction**,
and the two that matter for recipe choice bracket the ternary range as derived.
Ceilings revised above; the 350M → 760M ternary span and the "recipe decision,
not hardware limit" conclusion are unchanged.

Two things the measurement adds that the arithmetic did not:

1. **fp32 master weights cost ~17.7 B/param, not 16.** The extra ~1.7 is
   autocast's transient bf16 weight cache (~2 B/param). Any recipe holding fp32
   latent weights should budget 18, not 16.
2. **That configuration came within 0.09 GiB of OOM at 360M** — peak *reserved*
   was 7.41 GiB against 7.5 GiB usable. Not "roughly the ceiling": it *is* the
   ceiling, with no room for a longer sequence, a larger batch, or an eval pass
   sharing the device. Treat fp32-latent QAT above ~350M as out of reach on this
   card rather than merely tight.

Still computed rather than measured: the **ternary** rows specifically. The
probe exercises float full fine-tuning, so it validates the per-parameter
accounting and the optimizer/gradient terms, but not the materialised quantized
tensor a QAT recipe adds on top — which is why the QAT rows sit above their
float equivalents.

Method note: the first probe run returned `bytes_per_param: 0.0` because the
callback was constructed but never passed to `SFTTrainer(callbacks=[...])`. The
explicit `warning` field added before the run caught it immediately instead of
letting a plausible-looking zero into these notes. Worth keeping that habit — a
measurement harness needs a way to say "I did not actually measure this".

## 2026-08-07 — an unmerged LoRA adapter doubles generative eval cost

Noticed while the phase-0 evals ran. Identical task (IFEval, 541 prompts),
identical hardware, same 1200 MHz cap, `SW/HW Thermal Slowdown` both
**Not Active** on the slower run — so this is not thermals:

| run | s/it | total |
| --- | --- | --- |
| base SmolLM2-360M | 11.86 | 1 h 46 m |
| same + LoRA adapter via `peft=` | **22.46** | ~3 h 22 m projected |

**1.89× slower**, and the only variable is the unmerged adapter. Generative
decoding is memory-bound and sequential, so the extra per-layer LoRA matmuls
land on the critical path of every token rather than being absorbed into a
large matmul the way they are during training.

**Implication for phase 1:** the harness evaluates at *every stage boundary* of
a sequential fine-tuning run (§6.1a). Paying a 1.89× tax on generative evals at
every boundary would be a large fraction of the phase's compute budget. Merge
the adapter (`merge_and_unload`) into a throwaway copy before evaluating, or
stay on likelihood-based probes — which §6 already prefers for unrelated
reasons, and which this makes doubly attractive.

Not yet measured: whether the same 1.89× applies to loglikelihood tasks
(arc_easy/hellaswag). Those are batched and compute-bound rather than
decode-bound, so the penalty is likely much smaller — worth confirming before
generalising this to all evaluation.

## 2026-08-08 — phase-0 before/after eval (task 5, closes open item 1)

SmolLM2-360M base vs the same model with the 400-step LoRA SFT adapter.
`lm_eval` 0-shot, `--batch_size auto`. Base run 1 h 46 m, adapter run 3 h 22 m
(see the unmerged-adapter note above for why the second is slower).

| task / metric | base | +LoRA | Δ | Δ in SE |
| --- | --- | --- | --- | --- |
| arc_easy acc | 0.7029 | 0.7130 | +0.0101 | 0.8 |
| arc_easy acc_norm | 0.6814 | 0.6662 | −0.0152 | 1.1 |
| hellaswag acc | 0.4317 | 0.4361 | +0.0044 | 0.6 |
| hellaswag acc_norm | 0.5630 | 0.5686 | +0.0056 | 0.8 |
| ifeval inst_level_loose | 0.2830 | 0.2926 | +0.0096 | n/a |
| ifeval inst_level_strict | 0.2734 | 0.2830 | +0.0096 | n/a |
| ifeval prompt_level_loose | 0.1553 | 0.1516 | −0.0037 | 0.2 |
| ifeval prompt_level_strict | 0.1442 | 0.1479 | +0.0037 | 0.2 |

("Δ in SE" uses the naive independent-difference standard error,
`sqrt(se_base² + se_tuned²)`. The runs are paired on identical items, so the
true paired SE is smaller — but see the caveat below.)

**The honest read: nothing here moved.** Every delta sits at or under ~1.1
standard errors, including the one negative (`arc_easy acc_norm`). The
instruction-following metrics move in the direction you would hope for after
SFT on `smol-smoltalk` — inst-level +0.96pp on both loose and strict — but not
by enough to distinguish from noise.

**Yet the fine-tune plainly worked**: held-out eval loss 1.362 → 1.187 and
token accuracy 66.1% → 69.0% over the same 400 steps. So the loss-based probes
resolved a change that four benchmark accuracies could not.

**This is the phase-0 result that matters most**, because it is direct evidence
for a methodology decision the spec had already made on theoretical grounds
(§6: "likelihood-based probes remain primary — they stay sensitive where
accuracies floor or saturate"). At this model scale, with this size of
intervention, benchmark accuracy is simply not a usable instrument. A forgetting
experiment that reported only accuracies would have measured nothing at all.
Phase 1 should treat accuracy metrics as secondary reporting, not as the
signal being tracked.

**Caveat, and a phase-1 fix:** the runs were launched without `--log_samples`,
so there are no per-item outputs and a proper paired significance test cannot
be run retrospectively — the SE column above is the conservative unpaired
approximation. Add `--log_samples` to `scripts/eval.sh` before phase 1; paired
testing on identical items is substantially more sensitive and costs only disk.

## Open items — the live list

Closed:

- ~~1. Before/after eval table~~ — done 2026-08-08. No benchmark metric moved
  beyond ~1.1 SE; the loss probes did. See the eval section above.
- ~~3. Re-derive the §4 VRAM envelope~~ — done 2026-08-07. Outcome inverted the
  open item's assumption: the ternary ceiling is recipe-dependent, not
  card-limited.
- ~~4. Measure full-FT memory directly~~ — done 2026-08-08. Computed table
  validated within ~10%.

Open:

2. **Clock-cap A/B** (1000 vs 1200 MHz over ~50 steps each) — does a lower cap
   raise *average* throughput by avoiding boost→throttle oscillation?
   **Needs a design card before compute is spent (spec §3).**
5. **Does the unmerged-adapter 1.89× penalty also hit batched loglikelihood
   tasks?** Likely much smaller (compute-bound, not decode-bound). If it does,
   add adapter-merging to the eval wrapper.
6. **Add `--log_samples` to `scripts/eval.sh`** so paired significance tests are
   possible. Cheap, and phase 1 needs it — the phase-0 eval could not be tested
   properly for want of it.
7. **Measure the ternary QAT rows directly.** Everything measured so far is
   float training; the materialised-quantized-tensor term is still computed.
   Naturally folds into the 1c recipe shakedown at 135M.
