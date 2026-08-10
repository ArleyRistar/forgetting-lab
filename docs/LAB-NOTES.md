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
gradient checkpointing as the LoRA run. Components read directly off the live
optimizer, not inferred:

| configuration | weights | grads | optim | **total** | predicted | activations | resv/alloc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 + `adamw_torch` | 2.00 | 2.00 | 4.00 | **8.00** | 8 | 776 MiB | 1.13 |
| bf16 + `adamw_bnb_8bit` | 2.00 | 2.00 | 2.81 | **6.81** | 6 | 739 MiB | 1.08 |
| fp32 master + `adamw_torch` | 4.00 | 4.00 | 8.00 | **16.00** | 16 | 1481 MiB | 1.08 |

Two rows are *exactly* the predicted value, which confirms the accounting
method: bf16 params give bf16 AdamW states (torch allocates state
`zeros_like(p)`), and fp32 params give the classic 16 B/param. The two
deviations are the interesting part.

**1. 8-bit Adam saves less than the arithmetic says — 6.81 B/param, not 6.**
bitsandbytes keeps the **embedding's** optimizer state in fp32 rather than int8.
Check: 47.2M embedding params at 8 B + 314.6M at 2 B = 1,006,757,760 predicted
against 1,017,421,312 measured (1.1% out; the remainder is blockwise absmax
quantization state). This is **scale-dependent** — the 49k-vocab embedding is
13% of a 360M model but a far smaller fraction of a 1B one, so 8-bit Adam gets
closer to its nominal saving as models grow. Do not assume 6 B/param at 360M.

**2. fp32 activations are double the bf16 ones** — 1481 MiB vs 776 MiB — which
is autocast's transient bf16 weight cache: 776 + (361.8M × 2 B = 690 MiB)
= 1466 MiB predicted vs 1481 measured. So an fp32-latent recipe pays ~18
B/param all-in, not 16.

**3. Budget against `peak_reserved`, not `peak_allocated`.** The allocator holds
8–13% more than it hands out, and reserved is what actually occupies the card.
The fp32 run reserved 7.58 GiB and, with the 190 MiB context, used **99.1% of
the 7.66 GiB card**. That configuration is not "near" the ceiling at 360M; it is
*at* it, with nothing left for a longer sequence, a bigger batch, or a
concurrent eval.

Ceilings re-derived from measured components, budgeting reserved against 95% of
the card:

| configuration | B/param | ceiling |
| --- | --- | --- |
| bf16 + 8-bit Adam | 6.81 | **~920M** |
| bf16 + AdamW | 8.00 | **~740M** |
| fp32 master + AdamW | 16.00 | **~340M** |

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

## 2026-08-08 — TRACE data vendored and profiled (phase-1a task 1)

The spec says "TRACE's released datasets (one format, one loader)", which reads
like a `load_dataset()` call. It is not one. **TRACE has no HuggingFace
mirror** — searches for the benchmark return nothing and the obvious repo names
404. The data ships as a single Google Drive zip published with the Jan-2024
repo (`BeyonderXX/TRACE`, last pushed 2024-01-24).

Fetched 2026-08-08 via `scripts/fetch_trace.sh`:

- `TRACE-Benchmark.zip`, 77 MB
- `sha256 11152d50f8a093b1fc9c6c924ec207915d173b9e2d715396ec8f8a837e2668a8`
- archived to `~/archive/` as well as `data/` — that Drive link is now the
  provenance root of every phase-1/2 result and it has no mirror.

**Format holds up.** Every example is exactly `{"prompt": ..., "answer": ...}`,
two keys, no exceptions, across all nine sets. The "one format, one loader"
property the spec wanted is real once the archive is in hand.

**The archive has four variants the README does not mention:**
`LLM-CL-Benchmark_{500,1000,5000}` and `LLM-CL-Benchmark_Reasoning`. `_5000` is
the canonical training set. Two traps: the `_500` variant's 20Minuten directory
is **missing `eval.json`**, and held-out sizes vary from **41** (NumGLUE-cm) to
2000 (C-STANCE, Py150, ScienceQA). Pin the variant; clamp `n_eval` to what
exists and report the real count.

### Length profile — this is what picked the dev tasks

Prompt/answer characters over `_5000`. The `>1024 tok` column uses the usual
~4 chars/token approximation, so it is indicative, not exact:

| task | n | prompt med | prompt p95 | answer med | >1024 tok |
| --- | --- | --- | --- | --- | --- |
| C-STANCE | 5000 | 153 | 243 | 1 | 0% |
| FOMC | 5000 | 312 | 515 | 1 | 0% |
| NumGLUE-ds | 5000 | 138 | 212 | 2 | 0% |
| NumGLUE-cm | 5000 | 193 | 341 | 2 | 0% |
| ScienceQA | 5000 | 275 | 636 | 805 | 0% |
| Py150 | 5000 | 663 | 9107 | 32 | 13% |
| 20Minuten | 5000 | 2220 | 4417 | 261 | 8% |
| MeetingBank | 5000 | 5649 | 67994 | 338 | **58%** |
| Lima (replay) | 1030 | 96 | 759 | 1563 | 0% |

**MeetingBank is unusable at seq 1024** — 58% of examples exceed the window.
Truncating the majority of a summarization set whose target describes the whole
transcript makes the objective ill-posed, not merely hard. It was the intended
third dev task on length-blind reasoning and was dropped on these numbers.

Dev trio is therefore **FOMC → Py150 → ScienceQA** (Arley approved 3 tasks,
2026-08-08): classification → code → science QA, all English, all inside the
window, maximally separated in token distribution given those constraints.

Two consequences for the probe, both worth stating before any number exists:

1. **Truncate prompts from the left, never the answer.** Answer-token NLL is
   the measurement; an example whose answer was cut is corrupted, not noisy.
2. **Never average NLL across tasks.** The trio spans 1 answer token (FOMC's
   single letter) to ~200 (ScienceQA), so a cross-task mean would track
   whichever task is wordiest and would move when nothing changed. Each task is
   compared only against its own baseline — which is also what §9 requires for
   the ternary/float comparison later.

FOMC's one-token answer is a feature: that column is a clean classification
log-loss, strictly more sensitive than the accuracy phase 0 showed to be a dead
instrument at this scale.

## 2026-08-08 — clock-cap A/B: the hypothesis is refuted (closes open item 2)

`scripts/clock_ab.sh`, 150 steps/arm of full fine-tune (`mem_probe.py`, batch 4
x seq 1024, bf16, gradient checkpointing), cooldown between arms, analysed by
`scripts/clock_ab_report.py`. 150 steps rather than the 50 the open item
proposed, because steady state takes 10+ min on this chassis and 50 steps would
have measured mostly the cold regime — which is precisely the regime the result
turns out to hinge on.

**Hypothesis:** a *lower* SM clock cap raises *average* throughput by preventing
the boost -> overheat -> hard-throttle oscillation. **Result: no. 1000 MHz is
8.5% slower.**

| metric | 1200 MHz | 1000 MHz |
| --- | --- | --- |
| wall-clock, 150 steps | **666 s** | **723 s** |
| mean s/it overall | 4.45 | 4.82 |
| mean s/it, steps 2-50 | 4.00 | 4.52 |
| mean s/it, steps 100-150 | 5.04 | 5.32 |
| SM clock mean | 1041 MHz | 916 MHz |
| SM clock min | 660 MHz | 540 MHz |
| temp mean / max | 79.6 / 87 C | 80.8 / 87 C |
| power mean | 68.9 W | 63.2 W |

### Why — the cap does not prevent throttling, it only lowers the ceiling

Splitting each arm in half is what makes the mechanism legible:

| arm | first half | second half |
| --- | --- | --- |
| 1200 | 1196 MHz, sd 19, 72 C | 892 MHz, sd 99, **87 C** |
| 1000 | 1005 MHz, sd **0**, 75 C | 829 MHz, sd 118, **87 C** |

While cool, the 1000 MHz cap is held *exactly* — standard deviation 0. That is
genuinely the stable behaviour the hypothesis wanted, and if the run had been
50 steps long it would have looked like a win. Once heat-soaked, both arms hit
the same **87 C** wall and both throttle hard, and the capped arm throttles
*below its own cap* to 829 MHz. The throttle is temperature-driven and the cap
is not a temperature control. So capping only gives away speed during the phase
when the card could have run fast, and buys nothing during the phase when it
cannot.

Corollary worth keeping: **87 C is the operating point of this chassis under
sustained load regardless of clock cap.** Dropping the cap by 200 MHz cut mean
power 68.9 -> 63.2 W and moved equilibrium temperature not at all.

### The confound, and why it does not overturn this

Arm 2 started at **50 C**, arm 1 at **39 C** — the cooldown's 900 s deadline
expired before reaching its 45 C target. The bias is directional and handicaps
1000 MHz, which was flagged before the numbers were read.

It does not rescue the hypothesis. The soaked window (steps 100-150) is
temperature-matched by construction — both arms are pinned at 87 C — and
1000 MHz is still **5.6% slower** there (5.32 vs 5.04 s/it). The confound
distorts the cold phase, which is the phase that is not doing the work.

**Decision: keep the standing 1200 MHz cap.** No rerun; the temperature-matched
window already answers it. Fix the harness before reusing it: cool to a target
or *abort*, never silently proceed on timeout.

### Two side-observations

1. **The measured derate here is 1.26x, not 1.9x** (4.00 cold -> 5.04 soaked at
   1200 MHz), despite the arm ending pinned at 87 C. The 1.9x planning number
   in §4 came from a different workload. Do **not** relax the 1.9x budget on the
   strength of this — one 11-minute full-FT run is not the same measurement —
   but the two numbers need reconciling before they are trusted to the same
   precision. Logged rather than acted on.
2. Both arms returned identical memory components (8.00 B/param, `warning:
   null`), independently reproducing the 2026-08-08 memory probe on a different
   day and a different clock cap.

### Method note

The first version of the report script silently merged two tqdm bars — the
`Loading weights: 106/290` bar and the training bar — producing a "290-step"
arm with a *negative* mean s/it. Nonsense that obvious is harmless; the danger
is the same bug landing on numbers that merely look plausible. The parser now
separates bars by their total and takes the one with the most entries.

## 2026-08-08 — the boundary probe is ~7x cheaper than estimated (phase-1a task 3)

`scripts/probe_cost.py` against base SmolLM2-360M, 3 dev tasks x 200 held-out
examples, seq 1024, bf16, forward-only.

**Estimated 4 min per boundary for 8 tasks. Measured 12.7 s for 3 tasks** —
about 4.2 s/task, so ~34 s for all eight, not 4 minutes. The whole 4-boundary
dev run spends **under a minute** on probing. One IFEval run costs 1 h 50 m.
The instrument is effectively free, which is the phase-0 lesson paying off:
the sensitive measurement is also the cheap one.

The estimate is superseded, not "confirmed" — it was out by 7x and only ever
existed to be replaced (hard rule 3).

### Baseline — every later boundary is read against this

| task | NLL | token acc | n_tokens | prompts truncated |
| --- | --- | --- | --- | --- |
| FOMC | 5.1510 | **0.000** | 200 | 0 / 200 |
| Py150 | 1.6355 | 0.691 | 2799 | **50 / 200** |
| ScienceQA | 1.6353 | 0.620 | 43720 | 0 / 200 |

Two things to notice. **FOMC token accuracy is exactly zero** — the base model
never emits the right letter, because it is a base model with no instruction
format. NLL 5.15 against ln(49152) = 10.8 for uniform, so it is far from
ignorant, just not answering in the expected shape. That is a large amount of
headroom, which is good for the experiment.

**Py150 truncates 25% of prompts (50/200), not the 13% the character histogram
predicted.** Code tokenizes at fewer characters per token than prose, so a
char-based estimate understates it. The answers are untouched — that is the
invariant — but a quarter of Py150 prompts are seeing a left-truncated context.

### The probe is deterministic, but NLL depends on batch size

Two runs at batch 4 gave **bit-identical** NLL to six decimals. There is no
run-to-run noise floor to subtract.

But batch size changes the answer:

| | FOMC | Py150 | ScienceQA | alloc | reserved | time |
| --- | --- | --- | --- | --- | --- | --- |
| batch 4 | 5.150951 | 1.635523 | 1.635345 | 2325 MiB | 5970 MiB | 12.7 s |
| batch 2 | 5.144454 | 1.635955 | 1.635489 | 1548 MiB | 4954 MiB | 14.2 s |

FOMC moves **0.0065** between the two. The cause is bf16 reduction order
changing with padding width, and FOMC is the most exposed because it scores
only 200 tokens total (one per example) where ScienceQA averages 43,720.

**Consequence: `probe.batch_size` is part of the experiment and is now in the
config hash.** Lowering it mid-run to fit a bigger model would manufacture an
NLL shift of the same order as a real effect, and nothing downstream would show
it. Changing it now correctly invalidates resume.

### Fragmentation: reserved was 3.2x allocated until bucketing

First measurement: **2321 MiB allocated but 7424 MiB reserved** — 97% of the
card for a forward-only pass, and the log shows the allocator hitting an OOM
and recovering. The 8-13% reserved/allocated gap in the §4 notes does not hold
here.

Cause: length-sorted batching gives nearly every batch a distinct width, and
the caching allocator holds a block per distinct shape. Padding widths up to a
multiple of 128 cut reserved to **5970 MiB** (-1454 MiB) with no change to the
numbers. `probe_all` also calls `empty_cache()` on entry and exit, because in
the harness the probe runs immediately after training with that allocation
still cached.

Worth generalising: **when reserved runs far above allocated, suspect shape
diversity before suspecting a leak.**

## 2026-08-08 — harness performance notes (phase-1a tasks 4-5)

Three fixes found while building the stage loop, none of which changed a single
probe number — verified by re-running the baseline against the values recorded
earlier today and getting them back to six decimals.

**1. Prepare only what a stage consumes.** `load_task` tokenized all 5000 rows
and *then* selected, so a 2-step CPU test paid for 5000 examples. Selecting
first cut the test path from **8.4 s to 0.05 s** (168x) and a real 200-step
Py150 stage from 8.4 s to 3.7 s. Not a change of experiment: example order is a
deterministic hash of `(seed, index)`, so the first N of a 5000-row pool and a
pool of exactly N are the *same* N examples. Whole suite: 116 s -> 34.6 s.

**2. Pre-trim prompts by characters before tokenizing.** Py150 prompts reach
162k characters; tokenizing one in full to keep ~1024 tokens is pure waste.
Worth only ~8% on its own (9.1 -> 8.4 s) because the cap only bites on the
extreme tail, but it bounds the pathological case.

**3. A checkpoint directory existing is not a checkpoint.** A crash *during*
`save_model` leaves the directory present but unloadable, and resume then died
inside peft complaining about a missing `adapter_config.json` instead of just
retraining the stage — the wrong failure at 3 a.m. `checkpoint_ok()` now
validates the contents per mode, and an unloadable checkpoint is treated as no
checkpoint. Found because a test faked a checkpoint with `mkdir` and the
harness, correctly, tried to load it.

### Crash ordering, written down because it is easy to get backwards

A stage is marked DONE only once its **boundary probe is on disk**. Marking
DONE when training finishes would mean a crash in the gap between training and
probing loses that boundary forever: the stage is skipped on resume and nothing
ever raises. Training is still not repeated in that window, because the
checkpoint is recorded *before* the probe runs — so a resume finds the weights,
skips training, and goes straight to probing. Both halves have tests.

### Measured prompt truncation at seq 1024, over 3200 training examples

| task | truncated | of |
| --- | --- | --- |
| FOMC | 0 | 3200 |
| ScienceQA | 7 | 3200 |
| Py150 | **659** | 3200 (21%) |

Py150's 21% is consistent with the 25% seen on its held-out split, and well
above the 13% the character histogram predicted — code tokenizes at fewer
characters per token than prose. Answers are never truncated; this is prompt
context only.

## 2026-08-09 — phase-1a shakedown: the instrument reads (closes task 6)

3 stages x 200 LoRA steps, SmolLM2-360M, seed 0, FOMC -> Py150 -> ScienceQA,
1200 MHz cap. `configs/dev-3stage.yaml` at commit `408ff61`. Wall clock ~37 min
including a deliberate mid-run kill; **well under the 2-3 GPU-h budgeted**.
Supervisor exited `rc=0` on attempt 0 both times.

### The loss matrix — held-out answer-token NLL

| boundary | FOMC | Py150 | ScienceQA |
| --- | --- | --- | --- |
| baseline | 5.1510 | 1.6355 | 1.6353 |
| after FOMC | **1.0619** | 1.9303 | 1.7455 |
| after Py150 | 1.4537 | **0.8636** | 1.8220 |
| after ScienceQA | 1.7557 | 0.9690 | **0.7749** |

Bold is each task at its best, i.e. immediately after being trained. **I
expected little or no forgetting from three short LoRA stages and was wrong** —
the structure is textbook and unambiguous.

**Forgetting.** Each task gives ground back after training moves on:

| task | peak | final | forgot | as % of its own gain |
| --- | --- | --- | --- | --- |
| FOMC (trained 2 stages before the end) | 1.0619 | 1.7557 | **+0.6938** | 17.0% |
| Py150 (trained 1 stage before the end) | 0.8636 | 0.9690 | **+0.1054** | 13.7% |

The task trained longer ago forgot more — 6.6x more in absolute NLL. But note
the two normalisations disagree in strength: as a fraction of each task's own
gain it is 17.0% vs 13.7%, a far milder gradient, because FOMC's dynamic range
(baseline 5.15) dwarfs Py150's (1.64). **Two points do not establish a curve**,
and which normalisation is right is a real question for phase 2, not a detail.
Report both or the effect size is whatever the author chose.

**Negative forward transfer.** ScienceQA degrades monotonically while never
being a training target — 1.6353 -> 1.7455 -> 1.8220, i.e. +0.11 then +0.19 —
before dropping to 0.7749 when finally trained. Training on *anything* made an
untouched task worse. This is only visible because the harness probes every
task at every boundary; probing only trained tasks would have missed it, and it
is the cheapest half of the matrix to collect.

### Accuracy and NLL disagree — which is the whole argument for the instrument

| boundary | FOMC | Py150 | ScienceQA |
| --- | --- | --- | --- |
| baseline | 0.000 | 0.691 | 0.620 |
| after FOMC | 0.460 | 0.675 | **0.621** |
| after Py150 | 0.230 | 0.804 | 0.606 |
| after ScienceQA | 0.290 | 0.781 | 0.809 |

After the FOMC stage, **ScienceQA's NLL worsened by +0.1101 while its token
accuracy moved 0.620 -> 0.621 — nothing.** The model got measurably worse at
ScienceQA without changing which token it ranks first.

This is phase 0's conclusion reproduced inside a single run, and on a mechanism
rather than a statistic. Phase 0 showed accuracy was too *noisy* to resolve a
change. This shows something stronger: accuracy is the wrong **observable** —
the degradation lives in the probability mass, and argmax discards exactly that.
An accuracy-only harness would have reported "no effect" for a real one.

FOMC's accuracy is also non-monotonic (0.460 -> 0.230 -> 0.290) while its NLL
moves monotonically (1.0619 -> 1.4537 -> 1.7557). A monotonic underlying change
read through argmax comes back non-monotonic.

### FOMC's -4.09 is mostly format, not finance

Accuracy 0.000 -> 0.460 on a 3-way A/B/C task. The base model never emits a
bare answer letter at all, so most of that enormous NLL drop is learning the
*output shape*. Do not report it as a gain in task competence. (Note 0.460 is
argmax over the full 49k vocab, not a 3-way choice, so it is not directly
comparable to a 1/3 chance rate either.)

### Crash-resume verified on the real thing

Killed the tmux session at **step 53/200 of Py150**, with `checkpoint-50` on
disk, then restarted the supervisor. Both layers worked:

- **Outer (run state):** re-entered at stage 1. `probe-baseline.json` and
  `probe-after-0.json` had **identical mtimes and sha256** afterwards, and
  `stage-0-FOMC` was untouched — so no retraining and no recomputed boundary.
- **Inner (TRL checkpoint):** Py150 resumed at global_step 50 rather than 0.

**How that was verified matters.** transformers 5.x prints no "Continuing
training from checkpoint" banner, so grepping for it finds nothing and looks
exactly like a broken resume. The evidence is the step/time trace:

```
step  elapsed  s/step
   0        0     -
  51        4    0.08   <- data-skip to the resume point
  53       12    4.00   <- real training
```

A jump of 51 steps in 4 seconds is the fast-forward; 0.08 s/step is not
training. **Check the discontinuity, not the banner.**

### A flaw the resume test found, now fixed

The supervisor named logs `attempt-$attempt.log` with `attempt` resetting to 0
on every invocation — so **the restart silently overwrote the log of the crash
that caused it**, which is the one file a post-mortem most wants. Logs are now
stamped per invocation (`$STAMP-attempt-N.log`, plus a `latest.log` symlink) and
`SUPERVISOR-DONE` is appended rather than overwritten. Verified by running the
smoke config twice and getting two logs and two marker lines.

This is what a shakedown is for: the bug was in the recovery path, which is
exactly the code that only ever runs when something has already gone wrong.

### Cost

- Probing: **84.6 s for all 4 boundaries** (~21 s each, vs 12.7 s standalone —
  the difference is the LoRA adapter in the forward path). Against 1 h 50 m for
  a single IFEval, the full forgetting instrument is free.
- Step time varies ~4x by task: FOMC **1.23 s/step**, Py150 ~4.5-5.0,
  ScienceQA ~4.1. TRL pads to the batch's longest sequence, not to `max_length`,
  so short-answer tasks train much faster. **GPU-hour estimates must be
  per-task, not per-step-count** — the estimate I gave before launch was ~3x too
  pessimistic because it assumed the smoke test's full-length sequences.
- No probe warnings at any boundary: every cell measured what it claims.

### Caveat

**One seed.** The pattern is clean and the mechanisms are individually
plausible, but nothing here has an error bar, and the recency gradient rests on
two points. Phase 2 requires >=3 seeds on anything result-bearing (spec §7) and
that is not a formality — the normalisation ambiguity above could easily be
larger than the effect.

## 2026-08-09 — 2606.27634 protocol read; two Lima data traps (phase-1b prep)

Fetched and read [arXiv 2606.27634](https://arxiv.org/abs/2606.27634),
*Continual Learning for Sequential Personalization of Small Language Models: A
Stability Monitoring Analysis* (Paula, Kupssinskü & Barros). The spec picked it
as the calibration target sight-unseen; it is a better fit than expected —
**it uses TRACE**, so the data we already vendored is the data they used.

Their protocol, quoted rather than inferred:

- Models: Qwen 3.5 0.8B, **Llama 3.2 1B Instruct**, Gemma 3 1B IT (all ≤1B).
- Tasks: **FOMC → ScienceQA → NumGLUE-cm**, 500 train each — i.e. the **`_500`
  variant**, not the `_5000` we pinned. Reversed order also tested.
- LoRA r=8, α=16, dropout 0.05, no bias, target **`all-linear`**; AdamW **reset
  per task**; lr 5e-5; batch 2 × accum 8 = 16 effective; **1 epoch per task**;
  seq len 512.
- Metrics: ACC, BWT (`a_k,j − a_j,j`), FWT; stability via **KL from base**,
  entropy change, top-2 margin, all on a fixed reference set disjoint from every
  task's train and eval.
- Results: Qwen final acc **0.591 ± 0.012**, KL drift 0.300; Gemma **0.320 ±
  0.029**, KL peak 1.623 ± 0.157; **KL vs accuracy r = −0.497, p < 0.001**; KL
  ≈ **0.8** proposed as an order-independent instability threshold.

The calibration target is the **KL–accuracy relationship**, not an absolute
accuracy. Matching an accuracy on a different model would prove nothing;
reproducing the negative correlation is a claim about the phenomenon.

### Trap 1 — Lima's held-out splits are entirely empty

Lima is TRACE's replay set and the obvious reference-set candidate: disjoint
from every task by construction. But its `eval` and `test` splits are
**100% empty answers — all 300 rows in each, in both `_500` and `_5000`**. Only
`Lima/train` (1030 rows, median answer 1563 chars) carries content.

So `Lima/eval.json` — the natural reach for a held-out reference set — has
**zero scorable answer tokens**. Use `Lima/train`, held out and never trained on.

The probe already refuses to invent a number here: an empty answer contributes
no unmasked labels, `n_tokens` reaches 0, and it returns `warning: "zero answer
tokens scored; the NLL below is not a measurement"`. That warning field was
added on the strength of the memory probe's fake `0.0` back on 2026-08-08, and
this is the second time it has paid for itself. Phase 1b pins it as a
regression test against real data rather than a synthetic case.

### Trap 2 — NumGLUE-cm is their third task and has 41 eval examples

It carries the most forgetting signal in the sequence (trained last, probed
after everything) on the **smallest held-out set of any TRACE task** — 41 eval,
81 test. `n_eval` is already clamped and reported, so this needs reporting
discipline rather than code: quote `n` beside every NumGLUE-cm number.

### Consequence for the harness

Six things phase 1a hardcoded now need to be config: TRACE variant, task order,
LoRA hyperparameters (`all-linear` is a different adapted set than our seven
named modules), epochs-vs-steps, sequence length and learning rate. Plus one
genuinely new measurement — **KL from the base model on a reference set**, which
is the float-side analogue of phase 1d's ternary flip-fraction and is worth
building carefully for that reason alone. Under LoRA it needs no second copy of
the weights: `with model.disable_adapter():` gives the base distribution exactly,
at no extra VRAM.

Plan: `docs/superpowers/plans/2026-08-09-phase-1b-calibration.md`.

## 2026-08-09 — Llama 3.2 1B is gated; a mirror with an identical digest

Arley chose Llama 3.2 1B Instruct for the 1b calibration (2026-08-09), and no
design card, as with the 1a shakedown.

**`meta-llama/Llama-3.2-1B-Instruct` is `gated: manual`** and this box's HF
token is not approved: a hard 403 on the config, not a slow path. Manual
approval could take days and is not something the box can do for itself.

Resolved without waiting: **`unsloth/Llama-3.2-1B-Instruct`** is ungated and its
`model.safetensors` sha256 is **bit-identical** to Meta's published digest —

```
1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
```

— confirmed both from the official (gated) repo's public API metadata and by
recomputing the hash from the downloaded file. This is *stronger* provenance
than the usual "a mirror, probably fine": there is a cryptographic link to the
official release. Pin the digest and verify on load, because a mirror silently
ceasing to be byte-identical is the failure mode that matters.

Measured on the box:

| | |
| --- | --- |
| params | 1.236B (16 layers, hidden 2048) |
| vocab | **128256** — 2.6× SmolLM2-360M's 49152 |
| LoRA `all-linear` r=8 | 5.64M trainable, 0.456% |
| fwd, batch 2 × seq 512 | **2844 MiB reserved** of 7844 |

The vocab jump is the thing to watch. Probe activation memory is
logit-dominated and scales with `vocab × batch × seq`, so the probe batch size
that was comfortable at 49152 may not be at 128256 — and `probe.batch_size` is
in the config hash precisely because changing it changes the numbers. Expect to
set it deliberately for this model rather than inherit 4.

## 2026-08-09 — synthetic controls: a gate-design error, and a clean noise floor

Phase-1b task 5. Three runs on SmolLM2-360M, LoRA r=16, 400 steps/stage at
lr 5e-4, 50 nonsense keys mapped to one of 8 single-token values. That value set
puts an **analytic scale** on the NLL rather than a relative one:

| NLL | meaning |
| --- | --- |
| ~0 | perfect recall |
| **log(8) = 2.0794** | chance — the association is gone |
| ≫ 2.0794 | *confidently wrong* — a conflicting value was learned instead |

### Results

| arm | A memorised | A after stage B | forgetting |
| --- | --- | --- | --- |
| **conflict** (same keys, different values) | 0.0005 / 1.000 | 12.5178 / 0.000 | **+12.5173 NLL, −1.000 acc** |
| **disjoint** (no shared keys) | 0.0005 / 1.000 | 3.0861 / 0.340 | **+3.0855 NLL, −0.660 acc** |
| **null** (lr 1e-12, weights frozen) | 0.000744 / 1.000 | 0.000744 / 1.000 | **+0.000000** |

**Conflict behaves exactly as the logic demands.** A ends at NLL 12.52 — far
*above* chance — with accuracy 0.000, far *below* it. The model is not confused
about task A; it is confidently wrong, because it was taught a different value
for those same keys. The symmetry holds too: at the after-A boundary, task B
already sat at NLL 16.35 / acc 0.000, confidently wrong before ever being seen.

### The gate design was wrong, and it nearly cost us a real result

The disjoint arm was specified as the noise floor, on the reasoning that with no
shared keys "nothing forces interference, so zero forgetting is the analytically
known answer". **That reasoning conflates two different claims.** "A model with
spare capacity *can* hold both" is a statement about what is possible. It is not
a prediction about what SGD *does*. 400 steps at 5e-4 overwrite the adapter
weights encoding task A whether or not task B's keys collide with it.

So +3.0855 is not an artefact — it is **genuine catastrophic forgetting**, the
phenomenon this project exists to measure. The disjoint arm cannot separate
"the harness invents forgetting" from "forgetting really happened", because both
produce the same reading.

Consequence, had this gone unnoticed: phase 2 would have inherited a noise floor
of **+3.09 NLL** and discarded every real effect beneath it. That is larger than
most effects phase 2 is likely to find.

### The control that actually isolates the artefact

Run the *identical* structure — same task, same 400 steps, same
checkpoint/save/reload/re-probe cycle — at **lr 1e-12**, so total parameter
movement over the stage is ~1e-9 and no representable weight update occurs. Any
forgetting measured is then artefact by construction.

**Result: 0.000000, bit-identical to six decimals.** Task B's NLL is unchanged
too (7.869061 at both boundaries), confirming nothing moved in either direction
rather than moving and cancelling.

So:

- **The harness contributes nothing measurable.** Phase 2's forgetting noise
  floor is ~0; resolution is limited by seed variance, not by instrumentation.
- **Disjoint's +3.0855 is entirely real.** Learning 50 unrelated facts destroyed
  66% of 50 perfectly-memorised ones, with zero key overlap.
- The 2026-08-08 finding that the probe is bit-exact at fixed batch size is now
  confirmed **end-to-end through train/save/reload**, not just probe-to-probe.

### A discarded first attempt, and why

The first run used 100 steps at lr 2e-4 and produced a superficially perfect
result: conflict forgot +0.7213 NLL, disjoint +0.0892, an 8× separation in the
right direction. It was discarded because the **premise failed** — after its own
training stage, conflict task A sat at accuracy 0.300 and disjoint at 0.200,
against 0.125 chance and 1.000 for memorisation. Neither arm had learned the
associations; the NLL drop from ~6.5 to ~2.08 was the model learning the output
*format* ("answer with one letter"), not the key→value mapping.

Measuring how much a harness degrades a model that never learned anything is not
a noise floor. Worth stating plainly because the numbers looked like a pass —
**an 8× separation in the predicted direction, from a broken premise.**

### Incidental: negative forward transfer with no shared keys

In the disjoint arm, task B sat at NLL 7.39 / acc 0.080 after stage A, *worse*
than its 6.55 baseline, despite sharing no keys. Training on A makes the model
worse at unseen keys — presumably it now confidently pattern-matches them onto
memorised ones. Not what the gate turns on, but it is the same effect the 1a
shakedown saw on ScienceQA and is worth watching in phase 2.

### Gate verdict

- **Sees forgetting that is provably present:** PASS (conflict, +12.52).
- **Does not invent forgetting:** PASS (null, 0.000000) — and note this is the
  question the disjoint arm was *supposed* to answer and could not.

## 2026-08-09 — NLL conflates format acquisition with knowledge retention

Banked mid-calibration because it is analysis rather than data, and would
otherwise exist only in a chat log. From the three Llama 3.2 1B forward-order
seeds (phase-1b task 6).

**All three seeds disagree with themselves depending on the observable.**
BWT on accuracy is −0.0076 / −0.0064 / −0.0018 (slight forgetting); BWT on NLL
is +0.6560 / +0.5972 / +0.5900 (clear *improvement*). Consistent across seeds,
so this is structural, not noise.

Tracing it: **FOMC's NLL improved from 2.704 to 1.322 after training moved on
to other tasks, while its accuracy sat flat at 0.270 → 0.260.** The model became
much better calibrated on FOMC — assigning far more probability mass to the
answer letter — without changing which letter it ranks first any more often.

The likely mechanism is format. Training on ScienceQA and NumGLUE-cm teaches the
general shape "answer with one short token", which sharpens probability on the
answer position regardless of whether the answer is right.

### This complicates the case for our own instrument

Phase 0 and the 1a shakedown established that **accuracy is insensitive** — it
missed a real +0.110 NLL change on ScienceQA entirely — and I concluded NLL is
the better instrument. This adds the other half of the picture: **NLL is
sensitive to things a forgetting study may not care about.** An NLL improvement
can mean the model learned to *look* like it is answering, not that it retained
anything.

So the honest position is not "NLL good, accuracy bad". They measure different
things:

| | picks up | misses |
| --- | --- | --- |
| NLL | calibration, format acquisition, confidence | — conflates them with content |
| accuracy | whether the answer is right | changes in confidence entirely |

Neither alone is sufficient, which is the strongest argument yet for Arley's
2026-08-09 call to compute both and treat divergence as a finding. Where they
disagree, **the disagreement itself localises what changed**: NLL up with
accuracy flat means calibration/format moved and knowledge did not.

**For phase 2 this is a live hazard.** The ternary/float comparison measures
forgetting via held-out NLL. If a ternary model's conversion changes how sharply
it emits answer formats — which is plausible, since quantization affects
confidence — then part of any measured NLL difference would be format, not
forgetting. Reporting accuracy alongside is the cheapest available guard, and
the loss matrix already carries both.

## 2026-08-09 — the 2606.27634 code repo settles the measurement gap

Our calibration numbers came out 4–8× below theirs across KL, BWT and FWT, with
matching signs — a uniform scaling across independent metrics, which pointed at
definitions rather than a real difference. Rather than keep guessing, delegated
a search for their implementation. It exists:
**https://github.com/tspthomas/slm_stability_cl** (HEAD `801a3b3`, 2026-05-06).
The code is definitive where the paper is silent.

### 1. Their KL is ONE token position per example. Ours was ~300.

`src/stability.py::get_next_token_log_probs` takes the logits at the **last
non-padding position** — i.e. the next-token distribution immediately after the
rendered prompt, at the assistant generation-prompt position. No answer tokens
are involved:

```python
last_indices = attention_mask.sum(dim=1) - 1
next_token_logits = logits[batch_indices, last_indices, :].float()
return F.log_softmax(next_token_logits, dim=-1)
```

We average over every answer token — a median of ~300 per Lima example.
**Averaging across hundreds of mostly-unchanged tokens dilutes the divergence,
and that is almost certainly the entire 4–8× gap.**

Two things we had right: the direction is `KL(current ‖ base)`, and the base is
obtained via `model.disable_adapter()` rather than a second copy — the same
choice we made independently.

### 2. Their reference set is 48 examples carved from the tasks, not Lima

`scripts/build_reference_set.py` takes **20% of each task's `eval.json`**
(seed 33), combines and shuffles: **48 examples = 20 FOMC + 20 ScienceQA +
8 NumGLUE-cm** for the 500-variant the paper uses. Not a separate corpus at all.
Our Lima choice satisfies the paper's stated *property* (disjoint from all task
data) but is a different set. The paper never states N_R; only the code does.

### 3. "OP" is generative exact-match accuracy, not token accuracy

`src/evaluate.py::evaluate_accuracy` **generates** greedily (`max_new_tokens=256`,
`do_sample=False`), trims at the first EOS, then compares normalized strings:

```python
gold = normalize_answer(example["answer"], task_name)
pred = normalize_answer(prediction_text, task_name)
is_correct = pred == gold
```

`normalize_answer` reduces multiple-choice tasks to the option letter and
numeric tasks to a canonical number. OP is then the mean accuracy over **tasks
seen so far**, and BWT/FWT are the standard definitions on that matrix.

We compute per-token argmax accuracy. For FOMC and NumGLUE-cm (single-token
answers) the two coincide; **for ScienceQA they are entirely different
quantities**, which explains the BWT/FWT scale gap independently of the KL.

### 4. Protocol differences we had not caught

| | theirs | ours |
| --- | --- | --- |
| scoring split | **`test.json`** | `eval.json` |
| prompt | model chat template + per-task instruction prompts, `add_generation_prompt=True`, no system prompt | our own `<\|user\|>`/`<\|assistant\|>` tags |
| LR schedule | **none** (no scheduler, no warmup) | cosine + warmup |
| seeds | 33 / 42 / 123 | 0 / 1 / 2 |

Confirmed matching: LoRA r=8 α=16 dropout 0.05 bias none target `all-linear`;
AdamW created fresh per task and explicitly deleted after ("Drop optimizer state
after the task" — our fresh-trainer-per-stage achieves the same, and we asserted
it); lr 5e-5; batch 2 × grad-accum 8; 1 epoch; seq 512; task order; the
`_500` variant; and all three model choices.

### What this means

**We have not replicated their protocol.** We built something structurally
similar and measured it differently, so the 12-run calibration as configured
could not have validated against their numbers no matter how it came out. The
four completed runs are discarded rather than mixed across measurement regimes.

The tension to resolve before re-running: **phase 2 will use our instrument, not
theirs.** Calibrating a configuration we will not use validates the wrong thing.
The workable framing is a *replication mode* — match their protocol closely
enough to check the harness against published numbers, then run phase 2 in our
own configuration having established the harness is sound. What is being
validated is the harness, not the prompt format.

## 2026-08-09 — replication mode: accuracy replicates; an apparent 50-point collapse was the turn terminator

Phase-1b task 6, one run (`repl-llama-s33`, Llama 3.2 1B Instruct, their seed 33)
after rebuilding the harness to match 2606.27634's published implementation.

### Result table

| boundary | KL (ours) | KL (paper) | FOMC | ScienceQA | NumGLUE-cm |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0000 | — | 1.801 / 0.500 | 1.756 / 0.607 | 12.616 / 0.034 |
| after FOMC | 0.6849 | 0.199 | 0.518 / **0.765** | 1.783 / 0.606 | 11.336 / 0.028 |
| after ScienceQA | 0.7766 | 0.258 | 1.928 / **0.265** | 1.332 / 0.674 | 8.563 / 0.040 |
| after NumGLUE-cm | 3.3747 | 0.630 | 0.570 / **0.765** | 1.354 / 0.663 | 1.814 / 0.591 |

Cells are NLL / token accuracy; n = 100 / 97 / 81 on `test.json`.

### The finding: a 50-point "collapse" that was not forgetting

FOMC token accuracy runs **0.765 → 0.265 → 0.765**. Accuracy that halves and
then *fully recovers* cannot be forgetting — nothing restores destroyed
knowledge. The token counts settle it. FOMC scores 2 tokens per example under
paper-style labels (the answer letter and `<|eot_id|>`), so out of 200:

| boundary | correct | letters | terminators |
| --- | --- | --- | --- |
| after FOMC | 153 | **53** | ~100 |
| after ScienceQA | 53 | **53** | ~0 |
| after NumGLUE-cm | 153 | **53** | ~100 |

**Letter accuracy is constant at 53. The entire 50-point swing is the turn
terminator**, twice. ScienceQA's answers average 216 tokens and teach the model
to keep generating, so it stops predicting `<|eot_id|>` after a bare letter;
NumGLUE-cm's average 2.2 tokens and restore the habit.

FOMC's task knowledge never moved. The metric moved 50 points in each direction.

**Why this matters beyond the replication.** Sequential fine-tuning across tasks
with different answer lengths will *systematically* produce terminator collapse,
and any metric that scores the terminator reads it as catastrophic forgetting.
Their generative exact-match is largely immune — a model that answers "B" and
then rambles still normalises to "B". We were manufacturing forgetting they
would never see.

`content_acc` (accuracy excluding each example's final answer token) was added
for this and is the number their OP actually is. It returns `None` rather than a
substitute when the answer is a single token and excluding the terminator leaves
nothing to score — same discipline as the probe's `warning` field.

### Accuracy replicates

Stripping terminators: FOMC **0.53**, ScienceQA 0.663 (216 tokens/answer, so
dilution is negligible), NumGLUE-cm ~0.284 (104 of 176 correct, less 81
terminators, over 81 examples). Mean **≈0.49 against their final OP of 0.485**.
Per-stage, our FOMC 0.53 matches their post-FOMC OP of **0.530** to three
significant figures.

So the harness, given their protocol, trains their model to their reported
performance. That is what the calibration gate was for.

### What it took to get there, and how each was caught

| gap | ours before | how found |
| --- | --- | --- |
| KL direction `KL(base‖cur)` not `KL(cur‖base)` | — | reading their definition after our numbers came out 5–25× low |
| **KL scope**: ~300 answer tokens vs **one next-token position** | 0.0056 | their code; the single biggest factor |
| **completion-only loss**: we trained on prompt tokens too | 1.1205 → 0.6849 | *overshoot* — after fixing scope we were 5.6× too high, with entropy and margin moving far more than theirs |
| prompt style: our tags vs their chat template + task prompts | FOMC acc 0.000 → 0.500 at baseline | their code |
| scoring the turn terminator | 0.765 vs 0.53 | the impossible negative in a hand calculation |

The overshoot one is worth remembering as a method: **being wrong in the
opposite direction after a fix is diagnostic**, and it pointed at the objective
rather than the measurement before I had read the relevant file.

### Still open: KL is 5.4× high

Final KL 3.375 against their 0.630. This is now an isolated discrepancy on one
metric rather than a systematic difference — accuracy lands on their number, so
it is *not* that our training is more aggressive. Remaining candidates: the
specific rows drawn into the 20% reference carve (same composition, different
examples), or their KL being computed at a checkpoint we are not matching.
Unresolved; do not claim the KL replicates.

### Consequence for phase 1a's numbers

Phase 1a trained with **full-sequence loss**, not completion-only — every token
including the prompt. That is a legitimate choice but is not standard SFT, and it
drives more drift than completion-only does. The 1a shakedown's forgetting
magnitudes are real but reflect a more aggressive objective than the literature
default. `train.completion_only` is now a hashed config field defaulting to the
old behaviour, so nothing already recorded is silently invalidated; phase 2
should choose deliberately rather than inherit.

Also found: **Llama's chat template injects today's date** into an automatic
system block, so the rendered prompt — and every number from it — changes daily
unless pinned. That silently breaks "re-runnable from a commit hash". Now pinned
to a fixed date; their implementation has the same exposure.

## 2026-08-09 — phase-1b calibration verdict (SUPERSEDED — see the correction below)

Nine runs in replication mode (3 models × their seeds 33/42/123, paper task
order), matching 2606.27634's published implementation. `scripts/calib_report.py`.

### Per model, final checkpoint

| model | our KL | their KL | our acc | their acc |
| --- | --- | --- | --- | --- |
| gemma-3-1b-it | **11.63 ± 2.51** | 1.623 | 0.586 ± 0.019 | 0.320 |
| Llama-3.2-1B-Instruct | **3.48 ± 0.07** | 0.630 | 0.488 ± 0.009 | 0.485 |
| Qwen3.5-0.8B | **3.35 ± 0.10** | 0.300 | 0.717 ± 0.009 | 0.591 |

### What replicates

**The drift ordering, exactly: gemma > llama > qwen.** Three models, correct
rank order, on a metric whose absolute values are ~5× off. The *relative*
stability ranking transfers even though the scale does not — which is the more
transferable claim of the two and the one a different codebase can actually
check.

**Llama's accuracy, closely.** Final 0.488 vs their 0.485; post-FOMC 0.530 vs
0.530. Seed variance is tight (±0.009).

**Qwen best in both.** 0.717 vs their 0.591 — highest in each.

### What does not

**The KL→accuracy link, which is the paper's contribution.** Their central
result is r = −0.497, p < 0.001: models that drift more score worse. We get
**r = +0.296, p = 0.121** over 27 paired points — not significant, and not
negative.

The ordering table says the same thing more legibly:

```
ours by KL desc : gemma > llama > qwen
ours by acc asc : llama < gemma < qwen
```

Those two must agree if drift tracks damage. They do not. **Gemma drifts 3.3×
more than Llama and still scores 0.10 higher.** Their story is "gemma is
unstable — it drifts most and collapses to 0.320"; we reproduce the drift half
exactly and get no collapse at all.

So the disagreement is specific rather than general: it is entirely the
**gemma/llama swap on accuracy**. Their gemma is the worst model by a wide
margin; ours is comfortably mid-field.

### The honest verdict

**The calibration gate does not pass as specified**, and it should not be
recorded as passing. What we can say:

- The harness reproduces their *protocol* faithfully enough to hit Llama's
  accuracy to three decimals at two of three checkpoints.
- It reproduces their *relative drift ranking* across three models exactly.
- It does not reproduce the *relationship* those two are supposed to have.

That third point is the paper's actual claim, so this is a failure to replicate
the contribution while replicating much of the setup. Which is a more useful
outcome than either a clean pass or a total mismatch: it localises the
disagreement to one model's accuracy rather than leaving everything in doubt.

### Caveats that could account for the gap, in order of suspicion

1. **Our accuracy is teacher-forced content accuracy; theirs is generative
   exact-match after normalisation.** For gemma, whose answers we never
   generate, a model that would ramble in free generation can still score well
   under teacher forcing. This is the most likely explanation for gemma's
   inflated accuracy and is *not* something a likelihood probe can fix — it
   needs the generative eval we deliberately deferred.
2. **Absolute KL is ~5× theirs** (open item 12), unresolved. The ordering
   surviving a 5× scale error is reassuring but the scale error is real.
3. **NumGLUE-cm is scored under the `\boxed{}` prompt**, which instructs
   step-by-step reasoning while we score the bare answer immediately after the
   prompt. Content accuracy 0.242 vs token accuracy 0.591 on Llama. Their
   generative eval extracts from `\boxed{}`; ours cannot.

Given (1), **the gemma/llama swap is plausibly a metric artefact rather than a
real disagreement about the models.** Resolving it requires generative
exact-match, which is the deferred half of task 4.

### Seed variance is larger than two seeds suggested

Llama seed 123 gives **BWT +0.1040** where seeds 33 and 42 give −0.0063 and
−0.0052 — one seed in three flipping the sign of the headline forgetting metric.
An earlier ±0.0008 from two seeds badly understated this. Pooled BWT across nine
runs is −0.023 ± 0.064, i.e. **the standard deviation is ~3× the mean.**

Phase 2's ≥3-seed requirement (spec §7) is a floor, not a formality. Combined
with the null control's ~0 instrumentation noise, **seed variance is now
conclusively the binding constraint on what this rig can resolve.**

## 2026-08-09 — CORRECTION: the calibration gate PASSES; the failure was our metric

The verdict above was wrong and is superseded. It recorded phase 1b as failing
to replicate 2606.27634's central claim. Generative exact-match evaluation
(open item 14) shows the failure was **our accuracy metric**, not their result.

### The deciding measurement

Their accuracy is generative exact-match; ours was teacher-forced. Evaluating
all nine runs generatively — adapters merged, greedy decode, their normaliser:

| model | OP generative | OP teacher-forced | paper |
| --- | --- | --- | --- |
| gemma-3-1b-it | **0.301** | 0.586 | **0.320** |
| Llama-3.2-1B-Instruct | **0.480** | 0.488 | 0.485 |
| Qwen3.5-0.8B | 0.488 | 0.717 | 0.591 |

Teacher forcing inflated gemma from 0.301 to 0.586 — enough on its own to
invert its rank and destroy the correlation. Llama barely moved (0.480 vs
0.488), which is exactly why the disagreement looked localised rather than
systematic, and why it was worth chasing rather than reporting.

### Both orderings now agree, and both match theirs

```
by drift (desc)   : gemma > llama > qwen
by accuracy (asc) : gemma < llama < qwen
```

**Cross-model KL vs generative accuracy: r = −0.984, p < 0.00001** (n=9)
against their **r = −0.497, p < 0.001**. Same sign, stronger. Gemma drifts most
and scores worst; qwen drifts least and scores best. That is their claim,
reproduced on an independent implementation.

### What this does and does not establish

**Establishes:** the harness reproduces their protocol, their per-model
accuracies (gemma 0.301 vs 0.320, llama 0.480 vs 0.485), their relative drift
ranking, and the drift→damage relationship that is the paper's contribution.
The calibration gate passes.

**Does not establish:** that our absolute KL matches. It is still ~5× theirs
(open item 12). The *ordering* survives that scale error, which is the more
robust and more transferable claim, but the scale discrepancy is unexplained.

**Does not establish** that qwen's accuracy matches: 0.488 vs their 0.591, the
one model still 0.10 out. Gemma and llama both land within 0.02.

### The methodological lesson, which is the durable part

An entire verdict — recorded, committed and pushed — turned on a metric
difference that looked like a detail. Teacher-forced and generative accuracy
agreed to within 0.008 on llama and disagreed by 0.285 on gemma. **A metric can
agree with another on one model and disagree wildly on the next**, so validating
it on a single model proves nothing about the rest.

This is the second time tonight the same shape of error appeared: the turn
terminator made FOMC look like it lost 50 points when its content accuracy never
moved, and teacher forcing made gemma look mid-field when generatively it is
last. Both were the measurement, not the model. For phase 2, where the
comparison is ternary vs float rather than model vs model, the same hazard
applies directly — a metric that behaves differently on quantized models would
manufacture exactly the effect the project is looking for.

## 2026-08-09 — first 135M ternary shakedown failed: missing per-layer norm

Loss climbed **monotonically with lambda** — 2.75 at lambda 0, 3.36 at 0.5,
**11.75 at 1.0** — against ln(49152) = 10.8 for a uniform guess. So the fully
quantised model was worse than random.

The shape is diagnostic. A spike that then oscillates means optimizer
instability; a smooth climb tracking lambda means the quantised *function* is
broken and training is not recovering it. It was the latter.

**Cause: no normalisation inside BitLinear.** The recipe calls normalisation
before activation quantisation essential and I skipped it, reasoning that
SmolLM2's pre-norm blocks already normalise. They do — but only the *block*
input. `o_proj` receives raw attention output and `down_proj` receives the raw
SwiGLU product, and per-token absmax quantisation divides by the largest
activation in each token, so a wide-dynamic-range input wastes most of the int8
grid. Those two layers are exactly the ones the block norms do not cover.

**A test passed while testing the wrong thing.** `test_norm_precedes_the_quantised_path`
asserted the block has `input_layernorm` and `post_attention_layernorm`. Both
true, both irrelevant to whether each BitLinear normalises its own input. It was
written under a heading claiming to check the thing it did not check.

### Fix

Parameter-free RMS norm inside `BitLinear.forward`, **interpolated by lambda**:

```python
xn = x + lambda_ * (rms_norm(x) - x)
```

The recipe applies the norm unconditionally. Interpolating keeps lambda=0
bit-identical to the float layer, which phase 1c's premise requires — the float
weights are the initial latent weights, and if lambda=0 is not exactly the float
model then the conversion does not start where it claims to.

Parameter-free deliberately: a learnable norm would add randomly initialised
parameters, breaking both the unchanged-parameter-count invariant and that same
premise.

This is the restart spec §9 budgets for QAT fiddliness. Cost: ~25 min of 135M
compute.

## 2026-08-09 — 135M ternary shakedown: model collapses to chance and stays there

Two runs, both 1500 steps at seq 1024, effective batch 16 → **24.6M tokens**,
1.97 GPU-h each.

| step (lambda) | v1 no norm | v2 with norm |
| --- | --- | --- |
| 500 (0.5) | 3.355 | 6.608 |
| 1000 (1.0) | **11.75** | **10.59** |
| 1500 (1.0) | — (killed) | **10.744** |

Uniform over the 49152 vocab is **ln(49152) = 10.80**. So v2 ends at chance,
and the last four logs (10.698, 10.690, 10.665, 10.744) are **flat** — the model
is not diverging, it has settled into a degenerate solution and 500 steps at
full quantisation do not pull it out.

### The norm was not the problem

I diagnosed v1's failure as the missing per-layer normalisation and restarted.
It bought 11.75 → 10.59, which is marginal and still chance. Worse, through
warmup the norm made things consistently *worse* (6.61 vs 3.36 at lambda 0.5),
because partially-normalised activations are a scale the pretrained model has
never seen. And `activation_quant` already rescales after rounding, so its
conditioning benefit was smaller than I assumed.

I also killed v1 at exactly step 1000 — the instant lambda reached 1, the worst
point in the schedule — without seeing whether it recovered over its remaining
500 steps. That was premature. v2 shows what those steps look like: flat.

### The real error: warmup was scaled in steps, not tokens

The plan says, in as many words, to keep the 1000-step lambda warmup rather than
scale it down, because "it exists to stop the model collapsing when quantisation
switches on, and that risk does not shrink with a smaller budget."

That reasoning was wrong. Their 1000 steps are at **2M tokens/step = 2B tokens
of warmup**. Ours are at 16k tokens/step = **16M tokens**. We ramp quantisation
**125× faster in token terms** — the model gets 125× less data to adapt at each
level of lambda. The risk does not shrink with a smaller budget, but the
*defence* does, and I scaled the wrong quantity.

### The budget picture, stated plainly

| | tokens | vs recipe |
| --- | --- | --- |
| HF recipe | 10B | 1× |
| our 30 GPU-h budget | ~390M | 3.9% |
| this shakedown | 24.6M | **0.25%** |

A shakedown at 0.25% of the reference budget cannot demonstrate conversion
quality. What it *can* show is stability, and it does: no NaN, no divergence,
warmup applied to all 210 layers, `final_lambda` 1.0, `warmup_completed` true.
The machinery works. The question is whether the compute exists to use it.

### Decision needed

Whether phase 1c is affordable at all is Arley's call, not mine. The options:

1. **Retry with token-proportional warmup** — ramp lambda over ~50% of the run
   rather than a fixed 1000 steps. Addresses a real error above, costs ~2 GPU-h
   at 135M. Cheapest informative next step.
2. **Lower the learning rate.** 1e-4 comes from a recipe with a 125× larger
   batch; at our batch it may be driving the collapse to a degenerate solution.
3. **Go straight to 360M at the full 30 GPU-h** and accept that the pair will be
   badly under-converted. Spec §6 1c asks us to *measure and report* the
   conversion gap, and phase 2 compares each twin against its own
   post-conversion baseline (§9) — so a weak but *matched* pair may still answer
   the actual research question.
4. **Rent a cloud GPU** for the conversion only, as spec §4 permits when an
   approved experiment needs more than the box.

My read: (1) then (2) are cheap and address identified errors, and are worth
trying before concluding the budget is the binding constraint. But if both fail,
(3) is more defensible than it sounds — H1 asks whether flip-fraction predicts
forgetting better than parameter distance does, and that question does not
require a *good* ternary model, only a matched pair with a real latent
trajectory.

## 2026-08-09 — literature sweep after three failed shakedowns: we were missing distillation

Delegated a breadth sweep after v1/v2/v3 all collapsed. It found work that
changes both the recipe and how H1 has to be framed.

### The recipe is out of date, and the fix is distillation

**BitNet Distillation (arXiv 2510.13998, Microsoft)** — three stages to convert
an off-the-shelf FP model to 1.58-bit: SubLN insertion, a short continual
pre-training warm-up, then **dual distillation** (logits + MiniLM-style
attention-relation). Evaluated on Qwen3 0.6B/1.7B/4B — our exact scale.

**Ternary Mamba (arXiv 2606.18114, Jun 2026)** — QAT from a pretrained
checkpoint **plus KD** reaches 48.1% zero-shot on a 1.3B model in **102M tokens
/ 4 H100-hours**, against 150B tokens from scratch. A ~1000× reduction.

**102M tokens is inside our budget** (~390M at 30 GPU-h). Our three shakedowns
used 25–50M with **plain LM loss and no teacher**. Both papers that succeed at
this use distillation from the float model; we were doing the one thing neither
does.

Also **arXiv 2505.08823** — RMSNorm before every linear projection *plus a
gradual layer-wise quantisation schedule* matches KD pipelines. We added the
norm and quantise all 210 layers simultaneously; the layer-wise schedule is a
second lever we have not pulled.

### A named failure mode that is a cousin of our metric

Ternary Mamba reports **"zero-ratio collapse"** — an instability arising *only*
in QAT-from-pretrained, not from-scratch, caused by **learnable** quantisation
scales. Their fix is a non-learnable per-group absmean recomputed each forward.

Ours is non-learnable and recomputed each forward already, so that much is
right, but **theirs is per-group and ours is per-tensor**. Worth checking
whether our collapse is the same pathology at coarser granularity.

Related: **Tequila (2509.23809)** and **Signed-Zero Ternary (2508.05905)** both
identify the ternary **deadzone** as the core failure — weights parked at the
zero boundary get only noisy STE gradients and never escape. A high flip count
may be trapped weights rattling rather than learning, so phase 1d should report
zero-state occupancy alongside flip fraction and split flips by transition type
(±1↔0 versus +1↔−1).

### H1 needs reframing, and a better comparator

**Metaplasticity in binarised NNs (arXiv 2003.03533, Nature Comms 2021)** is the
closest prior art to our primary hypothesis: latent weight magnitude as a
flip-resistance variable gives EWC-quality continual learning with no importance
term. **Flips-as-forgetting is already established in binarised nets** — as a
*mechanism*, on permuted MNIST, at tiny scale. Our novelty has to be stated as
ternary rather than binary, LLM scale, and flips as a **predictive measurement**
rather than a consolidation mechanism. Must cite; anyone who knows this line
will find it immediately.

**RL's Razor (arXiv 2509.04259)** tests forgetting predictors head to head and
finds **L2 and spectral-norm distance correlate only weakly** — large shifts
with no forgetting, forgetting with small movement. Forward-KL to the base model
is what predicts. So benchmarking flip-fraction against L2 alone would be a
strawman; **KL-to-base must be a comparator in both arms**. Fortunately phase 1b
already built exactly that metric.

**"Accuracy is Not All You Need" (arXiv 2407.09141)** already uses **"%flips"**
to mean *prediction* flips — the fraction of eval samples where the compressed
model's top-1 differs. Naming collision: ours must be called **weight-state
flips** or **ternary state changes** or we will be misread.

**Quantization-permanent unlearning (arXiv 2605.15138)** finds per-parameter
unlearning updates sit **47–828× below the NF4 bin width**, so they never cross
a boundary and evaporate under compression. Direct quantitative precedent for
"only state changes count".

### The gaps are real

Explicitly searched and **not found**:

- **Forgetting in ternary/1.58-bit LLMs.** The quantised-CL literature is 4/8-bit
  PTQ + LoRA, unlearning-under-quantisation, or binarised MLPs on permuted
  MNIST. Nobody has run sequential tasks on a ternary LLM and measured
  forgetting.
- **Weight-state flip fraction as a *predictor* of forgetting.** Closest are BNN
  flip-flop ratio as an optimisation-stability diagnostic, metaplasticity's
  flip-resistance as a mechanism, and prediction-flips. Nobody correlates weight
  flip fraction with forgetting.
- **A data-matched float twin control.** No conversion paper runs the float
  control at matched token budget. That control is genuinely ours.

### The warning that matters most

The HF blog **itself** reports that after low-bit fine-tuning on TinyStories,
WikiText perplexity blew up — general knowledge lost. Other sources report
ternary conversion of Llama-3.2-1B degrading to incoherent output.

**Our sub-1B setting sits right at the edge where conversion is documented to
fail outright.** So the ternary arm's baseline capability has to be verified
*before* any continual-learning phase, or forgetting and failure-to-convert are
confounded — and every forgetting number would be measuring the wrong thing.

## 2026-08-09 — the shakedowns failed because latent weights were bf16, not fp32

An independent implementation review found a **measured bug-grade error**, and
it explains all three failures including why lowering the learning rate made
things worse.

### The bug

`convert.py` loaded the model in bf16 and trained with `bf16=True`, so latent
weights *and* Adam state were pure bf16 with no fp32 master. Every reference
keeps an fp32 master — nanotron defaults `accumulate_grad_in_fp32: true` and its
`FP32GradientAccumulator` makes "a fp32 copy of parameters during
initialization", stepping the masters and copying down.

Measured on our own SmolLM2-135M checkpoint (mean |w| = 0.148, median 0.119):
bf16 has 8 significant bits and round-to-nearest with no stochastic rounding, so
an Adam step of magnitude ≈ lr rounds to **no update at all** for

| lr | weights frozen per step |
| --- | --- |
| 1e-4 (v1, v2) | **85.4%** |
| 2e-5 (v3) | **96.3%** |

v3 was freezing roughly 24 of every 25 latent weights each step. That is exactly
a mechanism for "settles into a degenerate solution and stays flat" — and it
explains the most confusing observation of the night: **lowering the LR made it
worse, because a smaller step is more likely to round to zero.**

Fix: load fp32 and keep `bf16=True`, which is autocast mixed precision over fp32
weights — the reference setup. At 135M that is ~2.2 GiB, comfortable. At 360M it
is at the ceiling, so use 8-bit Adam there.

### What the collapse actually was

Reproduced on our code with **zero training**: SmolLM2-135M at λ=1 gives loss
**15.95** (λ=0.5: 14.48, λ=0: 1.84; uniform is 10.80). So conversion itself
destroys the model, as documented — the HF blog reports Llama3-8B conversion
"losses starting at approximately the same value of 13" as random weights, and
"the introduction of BitLinear layers overwhelms the model into losing all its
prior information."

That reframes v2: it pulled the model from 15.95 down to ~10.7, the unigram
floor. It **converted and partially recovered**, then had ~8M tokens at λ=1
where the recipe had ~8,000M — with 85% of its weights frozen.

The blog's own SmolLM-135M curves train *downward* under full quantisation. Flat
at chance is not what their 135M looks like, which points at the bf16 bug rather
than at the token budget.

### Other corrections from the review

- **The λ-interpolated norm was mine, not the recipe's.** Every reference applies
  the norm unconditionally and accepts a discontinuity at conversion. Ours meant
  warmup ran on partially-normalised activations no pretrained model has seen —
  consistent with v2 being worse than v1 at λ=0.5. Now always-on. The phase-1c
  premise survives: it is about the *weights* being the initial latents, and a
  norm does not change a weight.
- **Do not tune the λ schedule at this scale.** The blog: at 135M "the warmup
  quantization technique didn't yield as much improvement... the curves closely
  align."
- **LR should go up, not down.** Microsoft report 1-bit models want *higher* peak
  LR than fp16. v3's 2e-5 was the wrong direction twice over.
- **β2 = 0.95** (Microsoft's 1-bit setting; HF defaults 0.999), weight decay 0
  — large wd shrinks latent magnitudes and makes ternary weights flip too often
  — and grad clip 1.0.
- **Loss under 1-bit training is S-shaped**, so intermediate readings do not
  predict final performance. Relevant to how any shakedown curve gets read.

### Viability, restated

The recipe authors ran SmolLM-135M themselves and its ternary loss trained
normally, so conversion at this scale is not known-impossible. But the smallest
published *successful* conversion with quality numbers is Qwen3-0.6B, and it
needed SubLN + 10B tokens + distillation. Nothing published shows a decent
converted model below ~10B tokens at any scale.

"Decent" is not our bar. A matched pair that has left chance and acquired a real
latent trajectory is, and nothing published says that needs billions of tokens
once the optimizer can actually move the latents.

## 2026-08-09 — ternary conversion WORKS at 135M (shakedown v4, closes the 1c blocker)

`configs`: 2000 steps, seq 1024, effective batch 16 → **32.8M tokens**, 2.18
GPU-h. λ warmup over 20% (hits 1.0 at step 400), lr 1e-4, **fp32 latent
weights**, norm always-on, β2 0.95, wd 0, clip 1.0.

### The curve

```
  50:12.94  100: 8.10  150: 7.35  200: 6.97  250: 6.83  300: 6.73  350: 6.62
 400: 6.66 <- lambda = 1.0
 450: 6.65  500: 6.49  600: 6.38  700: 6.23  800: 6.19  900: 6.09 1000: 6.11
1100: 5.99 1200: 5.95 1300: 5.93 1400: 5.86 1500: 5.87 1600: 5.81 1700: 5.85
1800: 5.75 1900: 5.79 2000: **5.77**
```

Reference points on the same model: **float (λ=0) 1.84**, **uniform over the
49152 vocab 10.80**, **untrained at λ=1 15.95**.

### Against the three failures

| | at λ=1 | final |
| --- | --- | --- |
| v1 (bf16, no norm) | 11.75 | killed |
| v2 (bf16, interpolated norm) | 10.59 | 10.74 — flat at chance |
| v3 (bf16, lr 2e-5) | — | killed; predicted to fail |
| **v4 (fp32 latents)** | **6.66** | **5.77** |

**v4 barely notices the λ=1 transition** — 6.62 → 6.66 → 6.65 → 6.49 — where v1
and v2 were destroyed by it. Everything else about v4 was the same recipe.

### What actually fixed it

Latent weights were bf16 with no fp32 master. bf16 carries 8 significant bits
with round-to-nearest and no stochastic rounding, so an Adam step of magnitude
≈ lr rounds to **no update at all** for 85.4% of weights at lr 1e-4 and 96.3% at
2e-5 (measured on this checkpoint, mean |w| = 0.148). The model was freezing
roughly 24 of every 25 latents per step.

That also resolves the most confusing observation of the day: **lowering the
learning rate made things worse**, because a smaller step is more likely to
round away entirely. Every reference keeps an fp32 master — nanotron defaults
`accumulate_grad_in_fp32: true`.

Two other reversions, both off-recipe choices of mine: the λ-interpolated norm is
now always-on (as every reference does), and the LR went back **up** to 1e-4
rather than down.

### The conversion gap, reported not minimised

**5.77 ternary against 1.84 float.** That is a large gap and it is the expected
outcome at 32.8M tokens against the recipe's 10B — 0.33%. Spec §6 1c asks us to
*measure and report* it, and §9 has phase 2 compare each twin against **its own**
post-conversion baseline precisely so an absolute gap cannot confound the
forgetting comparison.

What matters for the project is not that the ternary model is good — it is that
it **left chance and acquired a real latent trajectory**, which is what phase 1d
measures weight-state flips along. That bar is now cleared.

If we later want the gap smaller, the published lever is logit distillation from
the float teacher (BitDistill 2510.13998; Ternary Mamba 2606.18114 reaches a
usable ternary model in 102M tokens, inside our budget).

### Cost note for the 360M pair

2.18 GPU-h for 32.8M tokens at 135M → **~15 GPU-h per 100M tokens at 360M**
scaling by parameters alone, so the 30 GPU-h budget buys roughly 200M tokens for
the pair, or 100M each. fp32 latents at 360M sit at the VRAM ceiling (16 B/param
= 5.8 GiB before activations), so the pair needs 8-bit Adam.

## 2026-08-09 — ternary QAT memory measured at 360M (closes open item 7)

`scripts/mem_probe.py --ternary`, which now converts to BitLinear before probing
so the materialised quantised tensor is actually in the graph. SmolLM2-360M,
λ=1.0, batch 4 × seq 1024, gradient checkpointing. 224 linears converted
(32 layers × 7).

| configuration | B/param | activations | peak reserved | % of 7.66 GiB card |
| --- | --- | --- | --- | --- |
| ternary QAT, fp32 latent + fp32 Adam | **16.00** | 1384 MiB | **7386 MiB** | **98.8%** |
| ternary QAT, fp32 latent + 8-bit Adam | **10.81** | 1320 MiB | **5314 MiB** | **71%** |
| *float* fp32 + AdamW (2026-08-08) | 16.00 | 1481 MiB | 7758 MiB | 99.1% |

### The §4 estimates were wrong in an interesting direction

Spec §4 estimated the ternary rows at **~18 B/param (fp32 Adam)** and **~9
(8-bit)**, on the reasoning that ternary QAT "adds the materialised quantized
tensor" on top of the float cost. Measured: **16.00 and 10.81** — the fp32 row is
*lower* than estimated and identical to float, and the 8-bit row is *higher*.

Both deviations have the same cause. The quantised weight tensor is created
inside the forward pass, so it is an **activation, not a parameter** — it never
touches bytes-per-param. And with gradient checkpointing it is recomputed rather
than held, so all 224 of them cost only what one layer's worth costs at a time.
Ternary activations came out *below* float's (1384 vs 1481 MiB), because the
quantised weights and int8-range activations are cheaper to hold than the
autocast bf16 weight cache the float run pays for.

The 8-bit row is higher than estimated for the reason already recorded on
2026-08-08: bitsandbytes keeps the embedding's optimizer state in fp32, and the
49k-vocab embedding is 13% of a 360M model. 10.81 here versus 6.81 for the float
8-bit run is the fp32 latent weights (4 B/param instead of 2) plus fp32 grads.

### Consequence for the 360M pair

fp32 latents with plain AdamW is **98.8% of the card** — no headroom for a longer
sequence, a bigger batch, or a concurrent probe. **8-bit Adam is not optional at
360M.**

**Correction, from the real run (2026-08-09):** the probe said 8-bit Adam would
sit at 71% (5572 MiB reserved). The actual 4000-step conversion runs at **7303
MiB, 93% of the card** — 1.7 GiB more than predicted. The probe runs 12 steps
against a pre-materialised dataset; the real job additionally holds the
streaming tokeniser's buffers and a longer allocator history. So the probe is a
good *lower bound* on a configuration's cost and a poor estimate of headroom.

Nothing failed — but "29% spare" was wrong and "7% spare" is the number. A
probe-derived headroom figure should be treated as optimistic by roughly this
margin when the real job streams its data.

Every row in the §4 table is now measured rather than computed.

## 2026-08-09 — 86.9% of SmolLM2-360M is ternarised (a failure mode we avoided)

A community sweep (r/LocalLLM and r/LocalLLaMA, via Wayback — Reddit blocks
direct access) turned up a hobbyist ternary run that failed for a reason worth
checking ourselves: at 4.3M params with a 50k vocab and `d_model=256`, **86% of
its training compute went to the softmax projection** and only 14% reached the
ternary core. Their fix was cutting the vocab to 10k and tying embeddings. A
"ternary model" can be mostly float, and the loss curve would not say so.

Measured on our checkpoint (CPU census, not arithmetic):

| component | params | share |
| --- | ---: | ---: |
| BitLinear weights (224 layers) | 314.6M | **86.9%** |
| embedding, tied with `lm_head` | 47.2M | 13.1% |
| total unique | 361.8M | |

`tie_word_embeddings = True`, so the head and embedding are one tensor counted
once — the 47.2M is not paid twice. Our ratio is the inverse of theirs because
`d_model=960` across 32 layers dwarfs a 49k vocab. **No action needed**, but the
number is now on record: when we report a forgetting result for "the ternary
model", 13.1% of it is float, and that fraction would grow at smaller scale.

Two claims from the same sweep worth pre-empting rather than discovering later:

1. **A QAT BitLinear is supposed to be slower and heavier than `nn.Linear`
   during training.** The most-cited "train BitNet from scratch" repo was
   publicly called out for a BitLinear that still dispatches to `F.linear` in
   bf16 — a fake. Ours has the same property by design (quantise in forward,
   float matmul, STE backward); the speedup lives in inference kernels we are
   not building. This is the honest answer when someone asks why conversion is
   not faster, and it is not a bug.
2. **"2-bit QAT might be better than 1.58-bit at small scale"** — an early
   experiment at 15.5M params found the gap to fp16 large enough to say so. Our
   135M and 360M runs sit above that scale, but the objection lands directly on
   phase 1c's absolute numbers and should be answered, not ignored.

Context for the eventual write-up: nothing in the archived record covers
catastrophic forgetting in low-bit models, no BitNet training hyperparameters
appear anywhere, and every 2026 ternary tool named is inference-only. The one
2026 low-bit release with a continued-learning feature (deepgrove's
Maple-Preview, 20B) drew exactly our question in a top **Reddit** comment — "how
much does it degrade base performance?" — and it went unanswered. Caveat: these
are anonymous posts without reproduced numbers, and Wayback only indexes what it
crawled, so absence means absent from the reachable archive.

**Two corrections to this entry, 2026-08-09, from a later sweep:**

1. **Maple-Preview may not be ternary.** The Reddit post title says "ternary-weight",
   but its HF config reads `bits: 2, group_size: 128, mode: affine` (lm_head 4-bit
   g64), found by a user in `deepgrove/maple-preview` discussion #3. Unresolved —
   possibly a 2-bit GGUF of a ternary base, possibly not ternary at all. **Do not
   cite it as a ternary release without checking the base model's own format**; a
   reader will correct it publicly.
2. **The "how much does it degrade" question is a Reddit comment, not an HF Hub
   one.** A second agent read all three ternary-model discussion tabs in full and
   did not find it there. Attribute it to r/LocalLLaMA or not at all.

## 2026-08-09 — the reference recipes, verbatim; and three challenges to our framing

Two sweeps (GitHub issues + OpenReview; arXiv/Semantic Scholar + HF Hub) found
what Reddit did not: the actual published hyperparameters. Every number below was
quoted from a source the agent read; arXiv IDs were verified to resolve to the
claimed titles (`https://export.arxiv.org/api/query?id_list=…` — note **https**,
plain http is blocked from this box, which silently returned empty for all eight
IDs on the first attempt and looked like eight fabricated citations).

### The recipes we are implicitly claiming to follow

Microsoft's `The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf` (in
`microsoft/unilm/bitnet`), Table 2:

| model | size | LR | weight decay | warmup | Adam β |
| --- | --- | --- | --- | --- | --- |
| BitNet b1.58 | 700M | 1.5e-3 → 1e-3 | 0.1 → 0 | 375 | (0.9, 0.95) |
| BitNet b1.58 | 1.3B–3.9B | 1.2e-3 → 8e-4 | 0.1 → 0 | 375 | (0.9, 0.95) |
| LLaMA (float) | 700M | 2.5e-4 | 0.1 | 375 | (0.9, 0.95) |
| LLaMA (float) | 1.3B–3B | 2.0e-4 | 0.1 | 375 | (0.9, 0.95) |

Batch **1M tokens**, **100B tokens**, seq 2048. Two things matter here:

- **The ternary LR is 6× the float LR at the same size**, and we are at 1e-4 —
  ~15× below their 700M ternary LR. This **reframes the bf16 finding** (see the
  2026-08-09 fp32-latents entry): at a 1M-token batch and lr 1.5e-3 the per-step
  update is orders of magnitude larger and bf16 would not round it away. Our
  result is real but conditional on *bf16 + small batch + lr 1e-4*. Report it as
  an update-survival rate over (dtype × LR), never as "bf16 latents are broken".
  **But do not let this reframing displace the cause.** One sweep concluded the
  low LR was "likely the real cause… more than bf16 is". That is wrong for our
  regime and we have the controlled experiment: we changed **dtype alone**,
  holding lr at 1e-4, and conversion started working — v4 at 135M and the 360M
  arm both. LR and dtype interact, and at Microsoft's LR the rounding would not
  bite; within *our* budget bf16 was the binding constraint. A plausible
  literature-based reframing does not outrank a measurement we ran.
- **They use a different LR per arm.** Nielsen et al. (below) explicitly use the
  *same* LR for both. So "same LR for both twins" is a defensible choice with a
  citation, but it is a choice, and it must be stated — otherwise a reviewer says
  the ternary arm was starved. Open item 26.

Nielsen, Schneider-Kamp & Galke, ACL 2025 Findings
(https://aclanthology.org/2025.findings-acl.694/, arXiv 2502.11895) — the closest
published experiment to ours, continued 16→1.58-bit QAT on OLMo-1B/Dolma:
AdamW, cosine+warmup, `learning_rate: 4.0e-4`, `weight_decay: 0.1`,
`betas: [0.9, 0.95]`, `t_warmup: 2000`, `precision: amp_bf16`, batch 4M tokens,
10k steps = 40B tokens. Their transition results (final loss):

| condition | loss |
| --- | ---: |
| 16-bit from scratch, 10k steps | 2.95 |
| continue from 2k 16-bit steps | **3.088** (best ternary) |
| continue from 4k | 3.097 |
| continue from 6k | 3.12 |
| full 1.58-bit from scratch | 3.15 |

Their recommendation: train 16-bit on **20–40%** of the data first, then quantise.
Converting a *fully* trained model — ours — sits past the end of that curve, which
is monotonically worse the later you switch. Worth acknowledging directly.

### Three challenges to our framing, in descending severity

1. **"Conversion wipes the prior information" — the challenge to the whole
   premise.** The HF blog (`1_58_llm_extreme_quantization`) reports pretrained
   Llama-3 weights (normal, std 0.013) and a random init (mixture, scales 50.25
   and 402) *"started at approximately the same value of 13"*, concluding **"the
   Llama 3 model loses all of its prior information when quantization is
   introduced."** Our own untrained-at-λ=1 loss of 15.95 on SmolLM2-135M agrees.
   If conversion already erases most of what was there, then "catastrophic
   forgetting in a ternary model" must be scoped to *what the converted model
   relearned*, not to the float model's original knowledge. This is the single
   most important framing decision in the project and it is Arley's call. It is
   also the honest answer to his earlier question of whether the premise is wrong:
   the premise is not wrong, but the baseline is the converted model, not the
   original.
2. **The STE gradient objection — the attack our mechanism will face.** Tequila
   (arXiv 2509.23809, ICLR 2026 sub. 9324) names **"deadzone trapping"**: weights
   stuck at the ternary boundary receiving only noisy gradients. Its reviewer
   SR3yFq8dQH (soundness 1) refutes it verbatim: *"if using a traditional STE
   estimation scheme, the calculation of the original weight gradient ∂L/∂wᵢ
   should be identical to the quantized weight gradient… when the original weight
   lies within the range (–δ, δ)… the weight gradient would be the same as outside
   the deadzone."* **Under plain STE, gradients do not distinguish the deadzone.**
   Any claim we make about near-threshold weights being special must therefore be
   about the *forward* contribution and the *moving boundary*, not gradient
   magnitude — and must be shown empirically, not argued. Our hypothesis survives
   this only in its decoupling form: |Δ effective| decouples from |Δ latent|.
3. **The unfair-comparison charge.** Spectra's reviewer Ta25rZvTRC (ICLR 2025
   sub. 11310): *"Both models were trained with 300B tokens, and the training loss
   does not appear to have fully converged… the FP model may require more tokens
   due to its larger model capacity."* At 66M tokens neither of our twins has
   converged. State the matching axis explicitly — identical tokens, identical
   order, identical steps — and name what is **not** matched: LR, convergence
   state, effective capacity.

### Numbers we can now compare against

- **Published per-step ternary flip rate: ~0.05%** (BitNet) and ~0.04% (ternary
  DQT), vs up to 8% for 8-bit — from "Direct Quantized Training with Stochastic
  Rounding" (arXiv 2412.04787) §5.2, measured at step 2000. This is a direct
  baseline for our headline metric, which otherwise reports into a vacuum.
  Same paper §5.3 partially undercuts the small-update story: suppressing the
  smallest 20% of updates had *"minimal impact"* on final loss at 130M.
- **>50% of BNN weights never change sign during training** ("silent weights",
  arXiv 2407.05257) — the binary-side anchor for a flip-rate result.
- **Expected zero fraction ≈ 1/3**: Microsoft reports the {-1,0,1} distribution is
  *"nearly uniform"*, and that raising γ to get more zeros *hurt* performance.
  A cheap sanity check on our absmean implementation.

### Tooling that exists and that we did not know about

- `tiiuae/onebitllms` (maintained, Triton kernels, `pip install onebitllms`) —
  a QAT BitLinear to validate ours against, plus `convert_to_bf16`. TII claim
  BitNet checkpoints revert to bf16 *"with minimal performance degradation"*,
  which if true is a cheap and striking experiment: revert our ternary twin to
  bf16 and see whether the forgetting signature survives.
- `schneiderkamplab/bitlinear` — the library used by the ACL 2025 paper above,
  so using it makes our numbers directly comparable to theirs.
- `huggingface/nanotron`'s 1.58-bit support is **an unmerged PR** (#180, still
  open) — the framework behind the famous HF result was never upstreamed.
- Do **not** use `kyegomez/BitNet`: open correctness bugs, including "Is
  activation actually quantized?" and "Expected BitLinear weight to be 1 or -1".

### Prior art: the verdict is that we are still novel, with must-cites

Nothing measures forgetting in a ternary LLM; nothing uses a data-matched float
twin as a *forgetting* control; nothing attributes forgetting to weight-state
flips under a moving threshold. Must cite, or a reviewer finds them immediately:

- **Laborieux et al., "Synaptic Metaplasticity in Binarized Neural Networks"**
  (arXiv 2003.03533; a second entry 2101.07592) — our mechanism, one bit lower,
  five years earlier: hidden real-valued weights as metaplastic variables, weights
  far from the threshold made to resist flipping, to reduce forgetting. Our
  differentiators: ternary not binary, LLM not vision, **measurement not
  mitigation**, and a data-matched control they lack.
- **Helwegen et al., "Latent Weights Do Not Exist"** (arXiv 1906.02107, NeurIPS
  2019) — latent weights are inertia, not weights; Bop optimises flip decisions
  directly. This is the licence for talking about weight-*state* flips at all.
- **"When Less is More: 8-bit Quantization Improves Continual Learning in LLMs"**
  (arXiv 2512.18934) — **points the opposite way to our intuition**: quantised
  models beat FP16 on retention, framed as quantisation noise acting as implicit
  regularisation. Not a duplicate (PTQ precision levels, no ternary, no twin, no
  flip mechanism) but we must engage with it or look uninformed.
- **Spectra / TriLM** (arXiv 2407.12327, 2506.23025) — already owns the
  *matched-precision-suite* idea (FloatLM and TriLM on the same data). Our twin is
  therefore **not** a novel control design; the novelty is applying it to
  forgetting. Their scaling result (TriLMs gain more from data than from
  parameters) is also the honest defence of our small token budget.
- **Tequila** (2509.23809) for "deadzone trapping" vocabulary; **TRACE**
  (2310.06762) for the benchmark; **Ternary Mamba** (2606.18114) for
  "zero-ratio collapse", a moving-threshold pathology in QAT-from-pretrained.
- Name-collision trap: "Ternary Feature Masks" (2001.08714) is continual learning
  with ternary *masks*, not ternary weights. Different thing.

Caveat on the whole prior-art section: abstracts only, no paper bodies read;
Semantic Scholar rate-limited to near-uselessness so coverage came from arXiv
keyword search, which matches title+abstract only. A paper doing this as a
secondary experiment would not have surfaced.

## 2026-08-09 — the ternary arm converts at 360M; the FLOAT twin is what OOMs

**Ternary 360M finished**: final loss **5.156** at step 4000, λ=1.0,
`warmup_completed: true`, 224 BitLinears, 65.5M tokens, **8h 24m** (30224 s),
7303 MiB steady. It crossed λ=1 at step 800 with no blip. Trajectory: 5.464 at
2000 → 5.235 at 3000 → 5.156 at 4000. Against the 135M run's final 5.767 and a
uniform-guess 10.80, the fp32-latent fix holds at 360M and gets better with size.

Then the float twin — the *cheaper* arm, by every intuition — **OOM'd on its
first backward pass** with the identical command. Measured, three configs, same
script, peak reserved:

| mode | micro-batch × accum | tokens/step | peak reserved |
| --- | --- | ---: | ---: |
| float | 4 × 4 | 16384 | **OOM** (768 MiB alloc failed) |
| float | 2 × 8 | 16384 | 5850 MiB |
| ternary | 4 × 4 | 16384 | 6774 MiB |

**The float twin costs more than the ternary twin at equal micro-batch.** The
cause is autocast's weight cache. Under `bf16=True` with fp32 master weights,
autocast caches a bf16 copy of each *leaf parameter* it casts. In float mode
`F.linear` receives `self.weight` directly, so all 314.6M BitLinear-eligible
weights get a cached bf16 copy (~600 MiB). In ternary mode `F.linear` receives
`w = ste(self.weight, weight_quant(self.weight), λ)` — a **computed tensor, not a
leaf parameter** — so the cache is bypassed entirely. The quantised path pays for
its extra intermediates and still comes out ahead.

Note the direction this cuts: the sweep found reviewers and users repeatedly
asking why BitNet training is not faster or lighter, and the honest answer is
that QAT is heavier. That remains true for *compute*. But for *memory under
autocast* the ternary path is genuinely lighter here, for an incidental reason
that has nothing to do with ternary weights being small — it is an artefact of
which tensor autocast decides to cache. Do not present it as a ternary benefit.

**Fix**: `--batch-size 2 --grad-accum 8`, product unchanged at 16384 tokens/step.
The dataset is a deterministic single-process generator, so the same 16 blocks
arrive per optimizer step in the same order whether that is 4×4 or 2×8, and with
fixed-length 1024 blocks the accumulated mean is identical up to float ordering.
The twins therefore still see the same tokens. Added `--expect-tokens-per-step`
as a hard assert, because "same tokens" is the entire point of the pair and it
was previously guaranteed only by remembering to type the same numbers.

Relaunched at 22:20 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(1.36 GiB was reserved-but-unallocated in the failing run). Live memory **6357
MiB**, against 6379 predicted from the probe plus the ternary arm's measured
+529 MiB probe-to-production delta — the calibration transferred.

**The twins are not identical in every respect, and the write-up must say so**:
they differ in micro-batch (2 vs 4), allocator config, and therefore step time.
They match on what the experiment needs: model, tokens, token order,
tokens/step, total steps, LR, schedule, optimizer, seed.

Two corrections to earlier planning numbers:

- The **§4 VRAM table's float row is optimistic** for this setup. It was measured
  without the autocast weight cache in play at this batch size.
- **Probe-to-production delta is +529 MiB for this probe**, not the +1.7 GiB
  recorded against `mem_probe.py`. The delta is a property of the probe, not of
  the box — a probe that includes gradient checkpointing, 8-bit Adam and real
  accumulation transfers far better. Prefer this style of probe.

## 2026-08-09 — three decisions Arley delegated, and a call to stop researching

Arley delegated these rather than deciding them himself ("you have more context").
Recording them as decisions so they are not re-litigated, with the reasoning, so
they can be **reversed on sight** if he disagrees — delegation is not agreement.

**1. Same learning rate in both twins (closes item 26).** Nielsen et al. (ACL
2025 Findings), the closest published analog to our design, state verbatim: *"We
employ the same learning rates for both the 16 and 1.58-bit baselines."*
Microsoft's 6×-higher ternary LR is for *from-scratch pretraining* at 1M-token
batches — a different regime from continued QAT off a pretrained checkpoint.
Decisive argument: different LRs would make the twins differ in **two** variables,
and the pair exists to isolate one. Caveat to state in the write-up: at 1e-4 the
ternary arm is probably under-trained relative to its own optimum, so absolute
quality comparisons are partly an LR artefact. We do not claim absolute quality,
so this costs us nothing we were trying to buy.

**2. `completion_only` — mask the prompt — for phase 2 (closes item 13).** It is
the literature default and 2606.27634 does it. The real reason: full-sequence
loss lets prompt-format drift contaminate the forgetting signal, which is the
exact artefact class that produced the phantom 50-point collapse in the
turn-terminator entry. A metric that moves because the answer's *shape* changed
is not measuring knowledge. Cost: phase-1a's numbers stop being directly
comparable, which is acceptable — 1a was a shakedown.

**3. Item 16 stays closed** (1c affordability, settled by v4 and the finished
360M arm).

**And the meta-decision: stop sweeping, start measuring.** Four literature
sweeps ran in one evening. They paid for themselves — item 21 caught a harness
that would have silently fine-tuned a *float* model and reported it as ternary,
and item 22 caught our headline metric being able to confirm H1 artefactually
through scale drift alone. But the marginal sweep had started returning citations
rather than decisions, and the open list went 20 → 31 while phase 2 did not start.

The rigour was accumulating faster than the results. For a project whose main
risk is publishing a wrong result that is the right direction to err, but it has
a failure mode of its own: never shipping. **Next after 1c closes is phase 1d
(weight-state flips)** — the cheapest real result, a direct test of the
hypothesis, and the one measurement the whole sweep confirmed nobody has made.
Items 21-30 get pulled in only where they are load-bearing for *that* number.

## 2026-08-09 — the "held-out" set was not proven held out (caught before it mattered)

`conversion_gap.py` drew its held-out blocks with `shuffle(seed=999)` against
training's `seed=0` and its docstring asserted that this made the split
"disjoint from what either twin saw". **That was false.** A buffered shuffle
permutes row order and shard order; it does not partition a corpus. Both streams
draw from the same `sample-10BT`, so held-out blocks could be blocks training had
already trained on, and the word "held-out" would have gone into the blog post
unearned.

This is the fourth instance of the project's recurring failure shape — *the
instrument silently measuring a different object than its name claims* (KL
direction, turn terminator, teacher forcing, and a test that asserted the block
has norms while the layer did not use them). It was caught by review before any
number was recorded, which is the first time this class has been caught *ahead*
of a result rather than after retracting one.

**Fix: make disjointness structural.** Use the *same* seed-0 stream training
used — reproducing its exact block order — and `.skip()` past everything it
consumed. The generator is deterministic and single-process, so every block after
the skip is provably unseen. A different seed would only have reshuffled the same
corpus, which is what caused the problem.

Sizing it on a measured quantity rather than an assumed one: training consumed
4000 x 16 = 64,000 blocks = 65.5M tokens, and FineWeb-edu averages **944.35
tokens/row** (measured, 400-row sample), so training needed ~**69,400 rows**.
`SKIP_ROWS = 250_000` is a ~3.6x margin. Raise it if the token budget grows.

The proof rests on one assumption, so it was **checked rather than assumed**:
two independent processes loading `sample-10BT` with `shuffle(seed=0,
buffer_size=10_000)` yield the same leading rows (verified 2026-08-09,
`seed 0 reproducible: True`). Without cross-process reproducibility, skipping
past training's rows would prove nothing — it would have replaced a false claim
with an unverified one.

**Also added WikiText-103 test as a second, out-of-distribution held-out set.**
It cannot overlap the FineWeb-edu stream by construction, and the HF 1.58-bit
blog reports WikiText perplexity (12.2 for their SmolLM-135M run), so it is the
number that makes us comparable to the closest published work. The script now
loads each model once and scores it on both corpora.

Note what the bug would and would not have broken: contamination would have
flattered both twins equally against the base model, so
`gap_ternary_vs_float_twin` — the number that matters, since both twins saw
identical tokens — would still have been fair. The damage was to the
*base*-model comparisons and to the honesty of the word "held-out".

## 2026-08-10 — zero fraction 0.3254 validates the absmean implementation

Item 30's kept half. Microsoft documents the {-1,0,+1} distribution in a trained
BitNet b1.58 as *"nearly uniform"*, i.e. a zero fraction near 1/3, and notes that
tuning the scale toward *more* zeros hurt performance. Measured on our converted
360M twin (loaded through `flab.loading`, so these are genuinely the effective
weights the forward pass uses, not latents):

| projection | zero fraction |
| --- | ---: |
| q_proj | 0.3538 |
| k_proj | 0.3500 |
| v_proj | 0.3341 |
| o_proj | 0.3302 |
| down_proj | 0.3212 |
| gate_proj | 0.3203 |
| up_proj | 0.3181 |
| **overall (224 layers)** | **0.3254** |

Against 0.3333 expected. This is an independent check on `weight_quant`: an
absmean threshold that was too aggressive or too slack would show up here as a
zero fraction far from a third, and it does not. The spread across projection
types (0.318–0.354) is small and in a sensible order — attention projections
slightly sparser than the FFN's.

Worth keeping for phase 1d: this is the *starting* zero occupancy. If
sequential fine-tuning moves it, that motion is itself part of the moving-threshold
story, and per-projection is the resolution at which to watch it.

## 2026-08-10 — PHASE 1C COMPLETE: the conversion gap is 2.59 nats

Both twins done, scored on identical held-out blocks through `flab.loading`
(224 BitLinear layers re-applied at λ=1, effective weights verified
three-valued). 64 blocks x 1024 tokens per corpus.

| model | FineWeb held-out | ppl | WikiText-103 test | ppl |
| --- | ---: | ---: | ---: | ---: |
| SmolLM2-360M base | 2.5257 | 12.50 | 2.5783 | 13.17 |
| float twin | 2.5373 | 12.65 | 2.6707 | 14.45 |
| **ternary twin** | **5.1288** | **168.82** | **6.2976** | **543.25** |

**The number that matters — `ternary_vs_float_twin` = 2.5915 nats** on the
in-distribution held-out set (13.3x worse perplexity), because both twins saw
*identical tokens in identical order*. Out of distribution it is **3.6268**
(37.6x). Base comparisons are documentation, not results: 2.6031 and 3.7193.

### What makes the twin worth having

**`float_twin_vs_base` is +0.0116 nats.** The float twin, given the same 66M
tokens the ternary arm got, ended up essentially where the base model started.
That is the control doing its job: it means the 2.59-nat ternary gap is
attributable to **ternarisation**, not to the extra training, the LR, the
schedule, or the corpus. Without this arm the gap would have been confounded and
we could not have said which.

Do **not** report that +0.0116 as "continued training made it slightly worse".
64 blocks is a small sample and that difference is well inside the range where
this eval cannot resolve direction. The 2.59-nat gap is not remotely in that
regime; a difference three orders of magnitude smaller is.

### Internal consistency (the check that the held-out set is sane)

| arm | final training loss | held-out loss |
| --- | ---: | ---: |
| ternary | 5.156 | 5.129 |
| float | 2.523 | 2.537 |

Both land on held-out essentially where they finished training, which is what a
single-pass 66M-token run with no repetition should do. It is also a check on the
new disjointness construction: contamination would have pushed held-out *below*
training loss, and it did not.

### The gap is worse out of distribution — a real finding

Ternary costs 2.59 nats on data resembling the conversion corpus and **3.63 on
WikiText**. The float twin shows the same asymmetry far more weakly (+0.012
in-distribution vs +0.092 OOD). So the converted model has partially re-fit to
FineWeb-edu rather than retaining general capability — consistent with the HF
blog's "loses all of its prior information when quantization is introduced" and
with the decision on item 31 to scope forgetting to *relearned* knowledge.

For phase 2 this is directly actionable: **probe tasks unlike FineWeb-edu will
start from a much weaker ternary baseline than in-distribution ones.** That is
exactly the item-20 gate, and this result says it will bite.

### Against the closest published run

The HF 1.58-bit blog reports WikiText perplexity **12.2** for a converted model
trained 5,000 steps on FineWeb-edu at a **2M-token batch** = ~10B tokens. Ours is
543 at 66M tokens — roughly **150x less data**, at a 16k-token batch. State the
comparison with those numbers attached or it is meaningless; our own notes are
not precise on which model that 12.2 refers to, so do not attribute it to a
specific scale without rechecking the source.

### What the twins do and do not share

Matched: model, tokens, token order, tokens/step (16384), total steps (4000),
LR (1e-4), cosine schedule, 8-bit Adam, β2 0.95, wd 0, clip 1.0, seed 0.
**Not** matched: micro-batch (2 vs 4) and allocator config, forced by the float
arm needing *more* memory than the ternary one under autocast, and consequently
step time (4.3 s vs 7.5 s) and wall clock (4h44m vs 8h24m).

Held-out disjointness is structural — same seed-0 stream as training, skipped
250,000 rows past the ~69,400 training consumed, cross-process stream
reproducibility verified.

**Phase 1c is complete.** The matched pair the project rests on exists and is
measured. Next is phase 1d (weight-state flips), which needs a design card.

## 2026-08-10 — phase 1d tasks 1-2: flips are weight-driven, and track L2 almost exactly

Instrument built (`src/flab/flips.py`, 23 tests on hand-built tensors) and run
over the conversion checkpoints of both twins at steps 0 → 1000 → 2000 → 3000 →
4000. Step 0 is the base model, whose weights *are* the initial latents. Zero GPU:
we already owned the checkpoints. Note `final/` is byte-identical to
`checkpoint-4000` (sha256 verified), so there is no fifth interval.

The float twin's flips are **counterfactual** — its weights are not ternary, but
applying the same state function answers a question H1 needs: how much would a
float model's would-be ternary states move under the same training?

| interval | arm | flip fraction | flips / L2 | scale-only | scale share |
| --- | --- | ---: | ---: | ---: | ---: |
| 0→1000 | ternary | 1.1788% | 0.000190 | 34 | 0.0009% |
| 1000→2000 | ternary | 1.0542% | 0.000186 | 107 | 0.0032% |
| 2000→3000 | ternary | 0.4856% | 0.000191 | 76 | 0.0050% |
| 3000→4000 | ternary | 0.0951% | 0.000208 | 25 | 0.0084% |
| 0→1000 | float | 0.9203% | 0.000199 | 18 | 0.0006% |
| 1000→2000 | float | 0.7286% | 0.000199 | 6 | 0.0003% |
| 2000→3000 | float | 0.3663% | 0.000199 | 2 | 0.0002% |
| 3000→4000 | float | 0.0776% | 0.000200 | 0 | 0.0000% |

(0→1000 spans the λ ramp, 0→800, so it is latent-state motion under partial
quantisation rather than fully-ternary motion. Both freeze conventions were
computed and give identical flip sets, as they must — the convention only
affects class assignment, not membership.)

### 1. The item-22 confound is ~zero here — but its share grows as training slows

**242 of 8,851,049 ternary flips (0.0027%) were scale-driven.** The worry that
motivated the whole decomposition — that a moving absmean threshold manufactures
flips and lets H1 confirm itself on a model that learned nothing — does not
materialise at 1000-step resolution. Flips are overwhelmingly weight-driven.

The guard was still worth building, and here is why: **the scale share rises
monotonically as weight motion shrinks**, 0.0009% → 0.0032% → 0.0050% → 0.0084%
across the cosine decay, roughly 10x from first interval to last. Scale motion is
small but roughly steady; weight motion decays. At phase-2 logging cadence —
which will be far finer than 1000 steps — the ratio could be entirely different.
**Do not carry "scale drift is negligible" forward as settled.** The dense burst
in task 3 is what tests it at per-step resolution.

### 2. Flip fraction is very nearly a linear function of L2 — the H1 problem

`flips / L2` is **0.000199–0.000200 for the float twin across every interval**,
and 0.000186–0.000208 for the ternary twin. Essentially constant, and essentially
the *same* constant for both arms.

This is what diffuse drift predicts: if weights move by roughly isotropic
increments of size σ, the fraction crossing a threshold goes as (density at the
threshold) × σ while L2 goes as σ√N, so the ratio is a constant set by the weight
distribution's shape near 0.5. Conversion training evidently looks like that.

**The consequence for H1 is direct.** H1 claims flip fraction predicts forgetting
*better than parameter distance*. If flips are a rescaled L2, they carry the same
information and cannot beat it. Under in-distribution continued training they are
a rescaled L2, to three significant figures.

That is not fatal, and it sharpens phase 2 rather than sinking it. The two
decouple only when weight motion stops being diffuse: many small moves near the
threshold give flips without much L2; a few large moves give L2 without many
flips. **So phase 2's real question is whether task shift produces structured
motion.** This result is the null it must beat, and it is now measured rather
than assumed — which is exactly what an instrument phase is for.

### 3. Ternary is not more fragile per unit of movement

Ternary logged 8.85M flips against float's 6.58M, but it also moved further (L2
62.0 vs 46.2 on the first interval). Per unit of L2 the ternary arm is **slightly
lower** than the float counterfactual for three of four intervals. Ternarisation
does not make weight states more volatile for a given amount of parameter motion;
the extra flips are bought with extra motion.

### 4. Ternary flips stick harder

Persistence (still in the new state one interval later): **ternary 0.866, float
0.781**; at two intervals, 0.824 vs 0.740. Consistent with the STE pushing
weights decisively across the boundary rather than leaving them oscillating.
Resolution caveat: an "interval" here is 1000 steps, and k=2 has one sample, so
spec §12 question 4 is settled mainly by the null arm.

### Incidental fix

`layer_delta` computed cosine in fp32 and returned **1.000073** — a value that
cannot exist. Accumulation error over millions of elements; now float64, with a
regression test. Worth noting because it would have gone into a plot unchallenged.

## 2026-08-10 — phase 1d task 3: the null floor, and a burst-placement mistake

Both twins continued on FineWeb-edu with **no distribution shift**, 400 steps,
`.skip(300_000)` (past training's ~69,400 and the held-out window's 250,000,
verified disjoint at runtime), λ asserted == 1 every step, weights-only saves.
Ternary 53.7 min, float 32.7 min, 26 checkpoints each. Plus a 120-step
**constant-LR probe** — see below for why that turned out to be necessary.

### The mistake, first: a cosine schedule has no steady state

The card put the per-step burst at steps 390–400 to escape LR warmup, after
review correctly caught that steps 1–10 would measure the warmup ramp. Both ends
are wrong. At 390–400 the cosine has decayed to ~0 and the per-step flip count
falls **109 → 83 → 74 → 66 → 47 → 36 → 24 → 11 → 5 → 0** — the final step has
literally zero flips because the learning rate is effectively zero.

Read off that tail, the per-step rate is 0.000015%, which is **~3000x below**
DQT's published 0.05% and would have fired the card's own "investigate if >10x
away" trigger for an entirely artefactual reason. The general lesson: *any*
per-step rate read off a cosine schedule is a rate at an unstated learning rate.

Fixed with a 120-step probe at `--lr-scheduler constant`, burst at 110–120, i.e.
after `warmup_steps=100`, so every burst step sits at exactly lr 1e-4. 15 minutes
of GPU. `null_arm.py`'s docstring now carries the reasoning; the previous version
asserted the opposite and would have misled the next person.

### The floor

**Per-step flip rate at lr 1e-4, no distribution shift: 0.008789%** — mean over
ten consecutive steps, range 0.00818–0.01024%, so it is a stable quantity rather
than a lucky sample. That is **5.7x below** DQT's ~0.05%/step, which is inside the
card's 10x band and in the pre-registered direction (they measured at step 2000 of
*from-scratch* training at a much higher LR; we continue an already-converged
model at 1e-4). **The investigation trigger does not fire.**

Flips track the learning rate closely — the 25-step series over the cosine arm
rises through warmup and decays with the schedule, peaking at 0.118% per 25 steps
at steps 100–125.

### Flips do not accumulate linearly — phase 2 cannot rescale by multiplying

Measured at constant lr 1e-4 from a single origin checkpoint:

| lag (steps) | flip fraction | linear prediction | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.010243% | 0.010243% | 1.000 |
| 2 | 0.019413% | 0.020487% | 0.948 |
| 5 | 0.041392% | 0.051217% | 0.808 |
| 10 | 0.067343% | 0.102434% | **0.657** |

A third of the linearly-extrapolated flips have disappeared by lag 10, because
weights re-flip and revert. **A per-step floor cannot be scaled to a per-interval
floor by multiplication**, and the curve is still falling at lag 10. Phase 2 must
either log at the cadence it reports, or use this curve.

### Persistence, at a stated cadence

Cadence-dependent, so the number is meaningless without one:

| cadence | k=1 | k=2 | k=4 |
| --- | ---: | ---: | ---: |
| 1 step (constant lr, null arm) | 0.9728 | 0.9460 | 0.8940 |
| 1000 steps (conversion, ternary) | 0.866 | 0.824 | — |
| 1000 steps (conversion, float counterfactual) | 0.781 | 0.740 | — |

This settles spec §12 open question 4's window-length half: **report persistence
with its cadence attached, and prefer the run's own logging cadence.** The
null-arm figure in `outputs/null/flips-ternary.json` mixes 25-step and 1-step
intervals and should not be quoted; the table above uses uniform cadences only.

### Item 22 is settled — and my extrapolation was backwards

Scale-driven flips, by resolution, ternary arm:

| interval | scale-only share of flips |
| --- | ---: |
| 1 step | **0%** (1 scale-only flip across 10 intervals) |
| 25 steps | ~0.001% |
| 1000 steps | 0.0009% → 0.0084% (rising as LR decays) |

In the tasks 1–2 entry I noted the share rising 10x across the conversion run and
warned it might be worse at finer cadence. **It is the opposite.** The share grows
with *interval length*: the absmean over 314.6M weights is extremely stable step
to step and only accumulates drift over long spans, while individual weights keep
crossing. At the resolution phase 2 will actually log, the confound is nil.

So the decomposition can be reported rather than relied on — with one live caveat:
this is measured under **no distribution shift**. A task that systematically
changes weight magnitudes would move the absmean more, and the instrument stays
worth running for exactly that case.

### flips ≈ 0.0002 x L2 survives at finer resolution

`flips_per_unit_l2` across the 25-step intervals is **0.000208–0.000249**
(ternary) and **0.000201–0.000202** (float), spanning a 20x range of L2. The
tasks 1–2 finding was not an artefact of 1000-step checkpoints. Flip fraction is
a rescaled L2 under diffuse drift, and that remains the null H1 must beat.

### Budget

GPU used in phase 1d so far: ternary null 0.9 h + float null 0.55 h +
constant-LR probe 0.25 h ≈ **1.7 of the 4.5 budgeted**. Task 4 (the item-20
capability gate) is the remainder. Disk at 22%.

## 2026-08-10 — phase 1d task 4: THE ITEM-20 GATE FAILS. Phase 2 is blocked.

Paired shuffled-answer control, derangement so no item keeps its own answer.
**n was 200 only for FOMC, ScienceQA and Py150**; NumGLUE-cm had 41 scorable
items and the synth tasks 50, because that is all their held-out splits hold. An
earlier version of this entry said "n=200" unqualified, which was wrong. Criterion: keep a probe only if (control − true) answer NLL ≥ 3 SE of the
paired difference. **The base model was run as a positive control for the gate
itself** — without it, "the twin scores 0.6 SE" cannot be distinguished from "the
instrument is broken", and this project has twice mistaken the second for the first.

| task | base | ternary twin | float twin |
| --- | ---: | ---: | ---: |
| ScienceQA | **6.1 SE** KEEP | 0.8 SE drop | **5.5 SE** KEEP |
| Py150 | **9.4 SE** KEEP | 0.1 SE drop | **9.2 SE** KEEP |
| NumGLUE-cm | 1.5 SE drop | 0.2 SE drop | 1.1 SE drop |
| FOMC | 0.1 SE drop | −0.8 SE drop | 0.1 SE drop |
| synth-conflict-a | 0.3 SE drop | −0.0 SE drop | 0.3 SE drop |
| synth-disjoint-a | 0.1 SE drop | −0.0 SE drop | 0.0 SE drop |

**Surviving probes for the ternary twin: none. The gate fails.**

**NumGLUE-cm is underpowered, not empty.** Base scores delta +0.566 at 1.5 SE on
41 items; the same effect at n=200 would land near 3.3 SE. It should be recorded
as "not enough held-out data to decide", not as "the base model cannot do it".
The ternary twin is at 0.2 SE there regardless.

### The instrument works, which is what makes the null readable

The base model discriminates strongly on two of six probes, so a 0.1 SE reading
is a statement about the model, not about the measurement. The gate also
discriminates between *tasks* — FOMC fails for **every** arm including base, so it
is a bad probe at this scale rather than evidence about any twin.

### The attribution is clean, and it is the twin's whole purpose

The float twin — same tokens, same order, same steps, same LR, same optimizer —
scores 5.5 and 9.2 SE against base's 6.1 and 9.4. It kept essentially all of it.
The ternary twin kept none. Absolute NLLs make the size of it plain:

| task | base | float twin | ternary twin |
| --- | ---: | ---: | ---: |
| Py150 answer NLL | 2.31 | 2.45 | **8.29** |
| ScienceQA answer NLL | 1.70 | 1.74 | **4.91** |

and the discrimination signal on Py150 collapses from +1.69 / +1.71 nats to
**+0.017** — a ~100x reduction. So this is not the continued training, the corpus,
the learning rate or the harness. **Ternarisation at 66M tokens destroyed the
task capability**, and we can say so because the control held.

This is consistent with everything else measured: the 2.59-nat conversion gap,
the 3.63-nat gap out of distribution, and the HF blog's "the model loses all of
its prior information when quantization is introduced".

### The synthetic tasks were mis-specified as candidates, by me

`synth-*` scores ~0 SE for the **base** model too — as it must, because those
associations are invented for this project and no pretrained model has seen them.
They are meant to be *taught* during phase 2 and then measured for forgetting.
The gate asks "can the model do this at t=0", which is the right question for
retained capability and the wrong one for a task the experiment teaches. Their
appearance in the card's candidate list was an error; **their failure here is not
evidence they are unusable.** Testing them needs a learnability check — train
briefly, then measure — which is a different and more expensive instrument.

### CORRECTION, same day: this does not block phase 2 on its own

The entry below concluded phase 2 was blocked. **That does not follow from this
gate**, and the contradiction is two paragraphs above it: the synth tasks are
*taught* by the experiment, not retained from pretraining, so their gate result
says nothing — and item 31 scoped forgetting to what the converted model
**relearned**.

The gate answered *can it recall?*. Phase 2 needs *can it learn?*. Nobody has
asked the second question. Card written 2026-08-10:
`plans/2026-08-10-phase-1e-learnability.md`, ~1 GPU-h, awaiting approval. Phase 2
stays paused until it reports — but the reason is "untested", not "impossible".

### What this means, per the pre-registered response

The card states: *"if no candidate probe passes Task 4, phase 2 cannot produce a
forgetting signal, and the response is distillation (item 17) or more conversion
tokens — not proceeding."* That condition has fired. **Phase 2 does not start.**

The gate did exactly the job it was added for: it cost ~0.5 GPU-h and it caught,
before seven phase-2 runs, that the ternary twin has nothing left to forget. A
flat forgetting curve from those runs would have been reported as "ternary models
forget less" — the most flattering possible misreading of a broken premise.

Note this does **not** invalidate tasks 1–3. The flip instrument, the null floor,
the sub-linear accumulation curve and the flips≈0.0002·L2 relation are all
properties of the conversion trajectory and stand on their own. What is blocked is
the forgetting experiment, not the instrumentation.

### Four things phase 1d promised and did not deliver

Recorded rather than quietly dropped:

1. **KL-to-base as a competing predictor was never computed.** The design card
   commits to it on both arms (Metrics, and change 5 of the v1→v2 changelog);
   `flip_report.py` emits L2 and cosine only. Item 19 is therefore still open in
   full, not half. Eval-only and cheap — compute it or descope it explicitly.
2. **Item 23's float-sliver covariate was never logged** either, despite the card
   saying "only its covariate is logged".
3. **Flips *and* L2 both exclude the 13.1% float sliver.** `flip_report.is_target`
   restricts to BitLinear tensors, so every `flips_per_unit_l2` in these entries
   is flips over *core* L2. Internally consistent, but a reader comparing against
   a whole-model parameter distance would be misled.
4. **The per-step floor 0.008789% is ternary-only.** No float constant-LR probe
   was run, so there is no float comparator at per-step resolution.

Also: "flips/L2 is constant to three significant figures" holds for the *float*
arm at 1000-step cadence; the ternary arm spans 0.000186–0.000249 across all
cadences, about ±15%. Show the scatter rather than asserting constancy.

### Phase 1d budget

Used ~2.2 of 4.5 GPU-h (null arms 1.45, constant-LR probe 0.25, gate ~0.5).

## 2026-08-10 — phase 1e: the ternary twin CAN learn. Phase 2 is unblocked.

Approved card `plans/2026-08-10-phase-1e-learnability.md`. Full fine-tune of each
twin on `synth-conflict-a` (50 nonsense keys, single-letter values), prompt masked
(item 13), lr 1e-4 (item 26), 300 steps, run **through `sequential.py`** — the
phase-2 harness, training a ternary model end to end for the first time.

| arm | held-out NLL before | after | token acc | derangement delta |
| --- | ---: | ---: | ---: | ---: |
| ternary | 7.2063 | **0.000057** | **1.000** | **+11.696 (17.7 SE)** |
| float | 6.1055 | **0.000036** | **1.000** | +13.933 (17.5 SE) |

Chance is `log(8) = 2.0794`; the pre-registered bar was NLL < 1.4 **and**
derangement ≥ 3 SE. Both twins clear both by orders of magnitude, on all 50
held-out keys. **Row 1 of the decision table fires: phase 2 proceeds on taught
tasks — no new conversion, no distillation.**

### The result worth stating plainly

**The ternary twin has no retained capability and full plasticity.** The item-20
gate found it could not discriminate on a single pretrained task (best 0.8 SE,
Py150 answer NLL 8.29 against float's 2.45). The same checkpoint, given 300 steps,
memorises fifty arbitrary key→value associations to perfect held-out accuracy and
an 11.7-nat margin over a deranged pairing.

Recall gone, learning intact. Those are separable, and conflating them is what
made "phase 2 is blocked" look true yesterday. The correct reading of phase 1d is
narrower than what I wrote: **conversion at 66M tokens destroyed what the model
knew, not its ability to learn.**

### Honest detail: the baseline is not a chance baseline

Before training, both twins sit *above* chance — 7.21 and 6.11 against 2.0794.
They are not knowledge-free at 2.08, they are **format-naive**: without training,
neither model knows to answer with a bare single letter, so it puts mass
elsewhere entirely. This does not weaken the result (the after-values are ~0), but
"before ≈ chance" would have been a wrong description and the gap is large enough
that a reader would notice.

### What this settles for phase 2's design

- **The task sequence must be taught tasks, not retained ones.** Both arms must
  run the same sequence for the pair to mean anything, and the ternary twin
  cannot do the TRACE tasks at all. The `synth-*` pairs were built in phase 1b as
  controls; they are now the primary instrument.
- That is a better fit than it sounds: `synth-conflict-a/b` share a key namespace,
  so learning B **must** destroy A — the forgetting is analytically bounded, not
  merely expected — while `synth-disjoint-a/b` gives the noise floor. Known
  answers on both ends is exactly what phase 1b built them for.
- **Item 17 (distillation) drops off the critical path.** It is no longer needed
  to unblock phase 2. It remains the route if we ever want a twin that is capable
  on *real* tasks, which is a different and more expensive goal.

### The harness survived its first ternary run

`assert_ternary` passed before and after training (224 layers, still three-valued),
λ stayed at 1, and `completion_only` was verified on a real example rather than
trusted — 1 supervised token of 36, exactly the answer letter. Item 21's wiring
works end to end, which was the second reason for routing this through
`sequential.py` rather than a fresh script.

### Budget

~0.3 of the 1.0 GPU-h carded. Training was 0.75–2.1 s/step at `max_length=256`,
the project's first short-sequence job — the 80% contingency was not needed.

## 2026-08-10 — an external USB fan has been running since ~2026-08-08

Arley added a small USB fan to the chassis around 2026-08-08, i.e. **before** the
360M conversion runs and everything measured since. So the project's planning
numbers already include it:

- thermal derate **1.9x**, steady-state **87 C**, ternary **7.5 s/step** at
  seq 1024, float **4.3 s/step** — all measured fan-on.

Two consequences. There is **no fan-off baseline**, so the fan's benefit cannot be
quantified retrospectively — a comparison against the 87 C figure would be
comparing fan-on to fan-on. And if the fan is ever removed, every derate-based
GPU-hour estimate in the cards becomes **optimistic**, so that would need
re-measuring rather than assuming.

The fan is a dumb VBUS load: it does not enumerate in `lsusb`, so it cannot be
switched from software without `uhubctl` and a hub supporting per-port power
switching (these are internal xHCI root hubs, which typically do not). Left
permanently on by decision — at ~2.5 W the saving from cycling it is negligible
against the cost of a bug that leaves it off during a thermally-limited run.

## Open items — the live list

Closed:

- ~~1. Before/after eval table~~ — done 2026-08-08. No benchmark metric moved
  beyond ~1.1 SE; the loss probes did. See the eval section above.
- ~~3. Re-derive the §4 VRAM envelope~~ — done 2026-08-07. Outcome inverted the
  open item's assumption: the ternary ceiling is recipe-dependent, not
  card-limited.
- ~~4. Measure full-FT memory directly~~ — done 2026-08-08. Computed table
  validated within ~10%.
- ~~2. Clock-cap A/B~~ — done 2026-08-08. **Hypothesis refuted**: 1000 MHz is
  8.5% slower overall and 5.6% slower in the temperature-matched soaked window.
  The cap does not prevent thermal throttling. Keep 1200 MHz.
- ~~6. Add `--log_samples` to `scripts/eval.sh`~~ — done 2026-08-08 in `89935a8`;
  the entry below was stale.

Open:

5. **Does the unmerged-adapter 1.89× penalty also hit batched loglikelihood
   tasks?** Likely much smaller (compute-bound, not decode-bound). If it does,
   add adapter-merging to the eval wrapper.
8. **Reconcile the two derate numbers** — §4 budgets 1.9x, the clock-cap A/B
   measured 1.26x over 11 min of full fine-tuning at 87 C. Different workloads;
   until reconciled, keep budgeting at 1.9x (the conservative one).
9. **Which forgetting normalisation?** Absolute NLL delta and percent-of-own-gain
   disagreed about the recency gradient's strength in the shakedown (6.6x vs
   1.2x). Decide before phase 2 reports an effect size, and report both.
10. **Re-examine the 1a shakedown's magnitudes now the floor is ~0.** Phase 1a
    reported +0.694 and +0.105 NLL forgetting with no artefact estimate. The null
    control says the harness adds nothing, so those are real — but they were
    one seed, and seed variance is now the binding constraint rather than
    instrumentation.
11. **Resume does not notice a code change.** `content_hash` covers the config,
    not the implementation, so editing a probe and re-running into an existing
    run directory silently skips completed stages and mixes results computed by
    two different versions. Hit on 2026-08-09 fixing the KL direction; handled
    by deleting the run dirs. `run.json` already records `git_commit` — the
    cheap fix is for resume to warn (not refuse) when it differs.
12. **KL is 5.4× above the paper's** after replication mode closed every other
    gap (3.375 vs 0.630 final, Llama). Accuracy matches, so it is not training
    intensity. Suspects: which rows land in the 20% reference carve, or a
    checkpoint-definition mismatch. Do not claim the KL replicates until settled.
- ~~13. Decide `completion_only` for phase 2.~~ — decided 2026-08-09 (delegated
  by Arley): **mask the prompt**. Full-sequence loss lets prompt-format drift
  contaminate the forgetting signal, the same artefact class as the turn
  terminator. See the delegated-decisions entry above.
- ~~14. Generative exact-match eval~~ — done 2026-08-09. Settled it: the swap was
  a teacher-forcing artefact, gemma scores 0.301 generatively against their
  0.320, and the calibration gate passes.
15. **Qwen's accuracy is 0.10 below theirs** (0.488 vs 0.591) where gemma and
    llama land within 0.02. Possibly the Qwen3.5 chat template's thinking mode,
    which their config disables via `enable_thinking=False` and which our
    renderer passes only when the template accepts the kwarg. Worth checking
    before phase 2 reuses this rendering path.
- ~~16. Is phase 1c affordable on this box?~~ — settled 2026-08-09. Shakedown v4
  reached 5.767 at 135M once latents were fp32, and the 360M arm is running
  inside the envelope. The blocker was the bf16 rounding bug, not the budget.
17. **Add distillation from the float teacher to the conversion.** Both recipes
    that succeed at our scale (BitDistill 2510.13998, Ternary Mamba 2606.18114)
    use KD; our three failed shakedowns used plain LM loss. Ternary Mamba
    reaches a usable model in 102M tokens, inside our budget.
18. **Rename our metric.** "%flips" already means prediction flips
    (2407.09141). Use "weight-state flips".
19. **Add KL-to-base as a forgetting predictor in both arms.** RL's Razor
    (2509.04259) shows L2 is a weak predictor, so flips-vs-L2 alone is a
    strawman. Phase 1b already built the metric.
20. **Verify the ternary twin's baseline capability before phase 2.** Conversion
    at sub-1B is documented to fail outright; without this check, forgetting and
    failure-to-convert are confounded. **Raised in priority 2026-08-09** — it is
    also the gate for item 23, and it is verbatim the one question the community
    asked of the only 2026 ternary continued-learning release. Run it straight
    after `conversion_gap.py`, before any phase-2 card.

The next four come from a 2026-08-09 review of the community sweep against the
spec, plan and code. All three code claims behind them were verified by grep
before recording: `BitLinear` appears in `bitlinear.py`, `convert.py`,
`conversion_gap.py` and `mem_probe.py` and **nowhere else**; `eval.sh:8`
hardcodes `pretrained=HuggingFaceTB/SmolLM2-360M`; `convert.py` sets
`requires_grad` nowhere, so the tied embedding is fully plastic.

- ~~21. Make the phase-2 harness BitLinear-aware, with a runtime assert.~~ —
  **BUILT 2026-08-09**, before any phase-2 number exists, which was the review's
  stated test of whether this project's error-catching has become preventive
  rather than reactive. `src/flab/loading.py` is now the only supported way to
  load a twin: `load_converted()` re-applies BitLinear at λ=1 when
  `is_ternary_checkpoint()` says so, then `assert_ternary()` verifies the
  *effective* weights take ≤3 distinct values and that every λ is exactly 1.
  Detection keys on the run's own `convert.json` (searched in the checkpoint dir
  and its parent, since weights live in `final/`), **not** on the directory name
  — a path called `ternary-360m` proves nothing. A ternary run whose warmup never
  completed raises rather than loading quietly, because its weights are not
  ternary whatever it is called. `conversion_gap.py` now goes through this path
  and exits if the ternary arm comes back with zero BitLinears. 11 tests in
  `tests/test_loading.py`, including the vacuous-pass case (a float model
  trivially satisfies "all BitLinears are ternary", so callers must check the
  count, not the absence of an exception) and a sabotaged quantiser. Verified
  end-to-end on the real checkpoint: 224 layers, guard passed.

  **Wiring completed 2026-08-10.** The harness turned out to have exactly *one*
  weight-loading site: `sequential.py::_load_base`. `probes.py`, `clmetrics.py`
  and `generative.py` all *receive* a model rather than opening one, so they
  needed no change — the earlier note that they "do not re-apply BitLinear" was
  true but misleading. `_load_base` now detects a ternary checkpoint, loads it
  through `flab.loading` **in fp32** (bf16 latents round ~85% of Adam updates to
  zero at lr 1e-4, and phase 2 fine-tunes these), and raises if a checkpoint that
  claims to be ternary comes back with zero BitLinear layers. Two more tests
  cover it, including an assertion on the dtype.

  Still outstanding: `eval.sh:8` hardcodes
  `pretrained=HuggingFaceTB/SmolLM2-360M` and cannot load a twin at all. It is
  not on the phase-2 critical path (the likelihood probes go through
  `sequential.py`), but it must not be pointed at a twin and believed.

  Original item, for the record:
21. **Make the phase-2 harness BitLinear-aware, with a runtime assert.**
    Checkpoints hold latent weights deliberately (`convert.py` docstring), so
    any loader that does not re-apply `bl.convert(..., lambda_=1.0)` scores a
    **float model and labels it ternary**. `conversion_gap.py:77-84` guards
    this; `sequential.py`, `probes.py`, `clmetrics.py` and `eval.sh` do not, and
    `eval.sh` cannot load the twin at all as written. Phase-2 *training* has the
    same exposure — sequential fine-tuning of the ternary twin must run through
    BitLinear at λ=1 or it silently fine-tunes a float model. Fix: one shared
    load path, plus an assertion that effective weights take ≤3 distinct values,
    run before any number is recorded. This is the same failure class as the
    turn terminator and teacher forcing — the metric measuring a different
    object than its name claims — and those cost us a retracted verdict each.
22. **Decompose flip fraction; raw flips can confirm H1 artefactually.** The
    absmean scale is per-tensor and recomputed every forward
    (`bitlinear.py:40-41`), so a shift in mean |w| reclassifies every
    near-threshold weight in a 921k–3.7M-weight tensor at once. Any fine-tuning
    moves mean |w|, so raw flip fraction partly measures *amount of training* —
    which correlates with forgetting trivially. Report scale-driven flips
    (recomputed holding the stage-start scale frozen) separately from
    latent-driven ones. If the moving threshold is the mechanism, that is a
    result; it cannot be folded invisibly into the headline number.
23. **[SPLIT] Log the float-sliver covariate now; DEFER the attribution
    experiment** until H1 shows signal — the hybrid-checkpoint swap is eval-only
    over checkpoints we already save, so it can run post hoc. The tied
    embedding *is* the output head, so the ternary model can adapt or forget
    with zero flips, and STE-noisy gradients in the ternary layers may bias
    optimisation toward routing adaptation through the clean-gradient float
    path — a confound correlated with the arm, not random noise. Spec §6 1d
    instruments flips for the ternary arm and L2 for the float twin, but nothing
    for the float components *of* the ternary model. Cheapest fix: log
    embedding/head L2 and KL-to-base per stage as a covariate. Real control:
    build hybrid checkpoints at each stage boundary (stage-k core + stage-(k−1)
    embedding, and the reverse) and score both — eval-only, reuses checkpoints
    we already save, and decomposes each stage's delta into flip-carried vs
    sliver-carried. Without it, H1's attribution is an assumption.
24. **Add a background flip-rate null arm.** At 66M tokens the ternary twin
    enters phase 2 still mid-conversion (loss falling 5.464 → 5.235 over the
    last 1000 steps) while the float twin sits near its pretrained optimum. Some
    phase-2 flips will therefore be continued conversion rather than task
    adaptation. Continue the twin briefly on more FineWeb-edu — no distribution
    shift — and report task-induced flips over that floor, as the phase-1b null
    control did for harness noise. This is the strongest reviewer objection to
    the eventual post, and it is answerable cheaply.
- ~~25. Pre-empt the 2-bit objection.~~ — **DROPPED to a write-up sentence**
  2026-08-09 (Arley accepted the review's cuts). Not an item: one sentence in the
  motivation section saying the claim comes from an unreproduced 15.5M-param
  experiment an order of magnitude below our scale, and that "is ternary the best
  low-bit point" is out of scope per spec §10.
- ~~26. Decide and state the LR policy across twins.~~ — decided 2026-08-09
  (delegated by Arley): **same LR in both arms**, following Nielsen et al., so the
  pair differs in one variable. Microsoft's 6× is from-scratch pretraining at 1M
  batches. See the delegated-decisions entry above.
- ~~27. Check the double-normalisation.~~ — **DROPPED to a write-up sentence**
  2026-08-09. The deviation is real (`convert.py` leaves the block's
  `input_layernorm`/`post_attention_layernorm` in place, so `q/k/v/gate/up` are
  normed twice and `o/down` only by ours), but both phase-2 arms are compared to
  their *own* baselines, so it cannot confound the forgetting result. Say "we
  retain the block norms rather than removing them as Microsoft's step 2
  specifies" and move on. No 135M ablation.
- ~~28. Ablate the λ warmup.~~ — **DROPPED** 2026-08-09. The conversion is
  finished and its warmup cost is sunk; whether warmup helped has no bearing on
  phase-2 validity. Cite the HF blog's measured "curves closely align" at this
  scale. Revisit only if we ever reconvert.
29. **[DEFERRED, with a trigger]** Run the 135M fp32-Adam vs 8-bit-Adam
    comparison only if measured flip rates come out far from the published
    ~0.05%/step, or if the H1 result is borderline. 8-bit Adam is unavoidable at
    360M and both arms share it, so it cannot differ between twins. The sweep found
    zero discussion of whether bitsandbytes 8-bit optimizer state interacts badly
    with latent weights near the absmean threshold — where quantisation error in
    the *optimizer* could flip effective weights. We depend on it for the 360M
    memory budget. At minimum, a 135M fp32-Adam vs 8-bit-Adam flip-rate
    comparison before flips become the headline metric.
- ~~30. Zero-fraction check.~~ — **DONE 2026-08-10**: 0.3254 overall vs
  Microsoft's ~0.3333, see the entry above. The baseline-comparison half remains
  a write-up table row, not a task.

30. [superseded, kept for context] **[SPLIT] Do the zero-fraction check now** (minutes, CPU — validates the
    absmean implementation against Microsoft's "nearly uniform" ≈1/3); the
    baseline comparison itself is a write-up table row, not a task.
    BitNet's measured
    per-step ternary flip rate is ~0.05% (DQT arXiv 2412.04787 §5.2); binary nets
    have >50% "silent weights" never flipping (2407.05257). Our numbers should be
    stated relative to these, not in isolation. Also check our zero fraction
    against Microsoft's "nearly uniform" ≈1/3.
- ~~31. Scope the forgetting claim to the converted baseline.~~ — **decided by
  Arley, 2026-08-09: we measure forgetting of what the converted model
  relearned**, not of the float model's original knowledge. Conversion may
  already erase most of the latter (HF blog: a pretrained model and a random init
  both start at loss ≈13 at λ=1; our own untrained-λ=1 is 15.95), so a claim about
  the original knowledge would be mostly measuring conversion damage.

  Consequences, all now binding on the phase-2 card:

  - **t=0 for every forgetting curve is the post-conversion checkpoint**, per twin.
    This is what spec §9 already specified; the decision makes it the *only*
    reading, and rules out any comparison against the original SmolLM2-360M as a
    forgetting number. `conversion_gap.py` reports that comparison separately and
    it stays documentation, not a result.
  - **Probes must be validated against the converted model, not the base model.**
    If the ternary twin is at chance on a task at t=0 it has nothing to forget,
    and a flat curve would mean "no capability", not "no forgetting" — a null we
    would misread in exactly the direction that flatters the harness. This makes
    item 20 a hard gate rather than a nice-to-have: measure the converted twin's
    capability on each candidate probe *first*, and drop probes where it starts
    at chance.
  - **The post's thesis is scoped in one clause** — "a converted, under-trained
    ternary pair at 360M, measured against its own post-conversion baseline" —
    and everything inside that scope can then be stated firmly.
  - The 2512.18934 result (quantised models *retain better* than FP16) is now
    directly comparable to ours in framing, since it also measures from the
    quantised model's own starting point rather than from a float ancestor.
