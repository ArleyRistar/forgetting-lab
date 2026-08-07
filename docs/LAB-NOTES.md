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
  batches, longer sequences, or a larger model are all affordable, and ternary
  QAT full fine-tuning (no LoRA available) has far more room than the
  350–560M ceiling previously estimated.
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
