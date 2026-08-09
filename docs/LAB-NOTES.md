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
360M**, and at 71% it leaves room to work in.

Every row in the §4 table is now measured rather than computed.

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
13. **Decide `completion_only` for phase 2.** Phase 1a used full-sequence loss;
    the literature default and 2606.27634 both mask the prompt. It changes how
    much a stage drifts, so it changes every forgetting number.
- ~~14. Generative exact-match eval~~ — done 2026-08-09. Settled it: the swap was
  a teacher-forcing artefact, gemma scores 0.301 generatively against their
  0.320, and the calibration gate passes.
15. **Qwen's accuracy is 0.10 below theirs** (0.488 vs 0.591) where gemma and
    llama land within 0.02. Possibly the Qwen3.5 chat template's thinking mode,
    which their config disables via `enable_thinking=False` and which our
    renderer passes only when the template accepts the kwarg. Worth checking
    before phase 2 reuses this rendering path.
16. **Is phase 1c affordable on this box?** The 135M shakedown collapsed to
    chance at 0.25% of the reference token budget and stayed there. Options and
    my read are in the 2026-08-09 shakedown entry; the call is Arley's.
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
    failure-to-convert are confounded.
