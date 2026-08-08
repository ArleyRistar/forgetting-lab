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
7. **Measure the ternary QAT rows directly.** Everything measured so far is
   float training; the materialised-quantized-tensor term is still computed.
   Naturally folds into the 1c recipe shakedown at 135M.
8. **Reconcile the two derate numbers** — §4 budgets 1.9x, the clock-cap A/B
   measured 1.26x over 11 min of full fine-tuning at 87 C. Different workloads;
   until reconciled, keep budgeting at 1.9x (the conservative one).
9. **Which forgetting normalisation?** Absolute NLL delta and percent-of-own-gain
   disagreed about the recency gradient's strength in the shakedown (6.6x vs
   1.2x). Decide before phase 2 reports an effect size, and report both.
