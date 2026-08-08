# Phase 1a — Sequential Fine-Tuning Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the instrument the §1 question is asked with — a sequential
fine-tuning rig (base → task A → task B → …) that measures forgetting with
likelihood probes, survives a 3 a.m. crash without human attention, and is
re-runnable from a commit hash. Spec §6 1a.

**Architecture:** Extend the `flab` package with four new modules and one
supervisor script. A run is a declarative YAML config; the harness walks its
stages, training with TRL and probing held-out NLL at every boundary, recording
everything into a run directory whose state file makes restart-at-stage-k
possible. Nothing in it is LoRA-specific or float-specific — phase 1c runs a
full fine-tune of a ternary model through the same code path.

**Tech Stack:** unchanged from phase 0 (Python 3.12, uv, torch, transformers 5.x,
TRL, peft, datasets). Adds `pyyaml`. No new GPU-side dependencies.

**Precondition:** phase 0 complete (it is). The clock-cap A/B must have finished
before any GPU step here — never two GPU jobs at once (CLAUDE.md).

## Global Constraints

- bf16 + gradient checkpointing on all training (spec §4).
- Development runs: **SmolLM2-360M, seed 0, single model** (spec §6 1a).
- **Three tasks, not eight** (Arley, 2026-08-08). All nine TRACE sets are still
  vendored in task 1 — fetching is one download and the archive is the
  provenance root — but the run configs, the loss matrix and the shakedown use
  a 3-task subset. Widening to 8 is a config change, not a code change, and is
  revisited after 1b's calibration gate passes.
- **Likelihood probes are the instrument.** Phase 0 measured benchmark accuracy
  moving < 1.1 SE while held-out loss moved clearly. Accuracies are secondary
  reporting only; they do not gate anything in this harness.
- **No unmerged adapter ever reaches a generative eval** — measured 1.89× tax.
- `outputs/` is never committed; code, configs and `uv.lock` are.
- Long runs in `tmux` on the box, with a completion marker that means
  *finished*, not *worked*.
- GPU-hour estimates use the **1.9× thermal derate**.
- Commits go straight to `main`.

## Scope boundary — what this plan is NOT

Named here so the seams are deliberate rather than discovered later:

| out of scope | belongs to | seam left in this plan |
| --- | --- | --- |
| ternary conversion / BitLinear | 1c | `mode: full` training path, exercised but on float |
| flip fraction, persistence, threshold histograms | 1d | `probes.py` returns a dict; 1d adds keys |
| replicating 2606.27634 | 1b | a run config, not new code |
| synthetic forgetting control | 1b | a task entry in the TRACE loader's registry |
| ≥3 seeds, result-bearing runs | phase 2 | `seed` is already a config field |

---

### Task 1: Vendor the TRACE data

**Do this first.** The entire phase-1a data plan rests on a Google Drive link
published in January 2024. It resolved on 2026-08-08 (`TRACE-Benchmark.zip`,
verified live), but it is a single point of failure with no mirror, and if it
has gone by the time we need it, every downstream task changes. Find out now.

**Files:**
- Create: `scripts/fetch_trace.sh`
- Create: `src/flab/trace.py`
- Create: `tests/test_trace.py`
- Modify: `docs/LAB-NOTES.md`

**Interfaces:**
- Produces: `data/trace/<task>/{train,eval,test}.json` on disk (gitignored);
  `flab.trace.load_task(name)` returning a `DatasetDict`.

**Background (verified 2026-08-08, not assumed):**
- TRACE lives at GitHub `BeyonderXX/TRACE` (102 stars, last pushed 2024-01-24).
  There is **no HuggingFace mirror** — searches for the benchmark and for a
  single-repo upload return nothing; the obvious repo names 404.
- Data ships as one Drive zip, *not* `load_dataset()`-able.
- The released format is genuinely uniform, which is what spec §6 1a wanted
  from "one format, one loader": per task, `train.json` / `eval.json` /
  `test.json`, each a JSON list of `{"prompt": ..., "answer": ...}` — confirmed
  by inspection, exactly two keys, no exceptions.
- Nine sets: eight training tasks (C-STANCE, FOMC, MeetingBank, Py150,
  ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten) plus Lima as a replay set.
- **The archive contains four variants**, which the README does not mention:
  `LLM-CL-Benchmark_{500,1000,5000}` and `LLM-CL-Benchmark_Reasoning`. Use
  **`_5000`** — it is the canonical TRACE training set (5000 train per task).
  The `_500` variant is a subset whose 20Minuten directory is also **missing
  `eval.json`**, so a loader that globs blindly across variants will break.
  Pin the variant in the loader; do not auto-discover it.
- Held-out sizes vary enormously by task (NumGLUE-cm has **41** eval examples,
  C-STANCE/Py150/ScienceQA have 2000). Any `n_eval` the config asks for must be
  clamped to what exists, and the actual count reported — a probe silently
  running on 41 examples where 200 were requested is a number nobody can trust.

- [ ] **Step 1: Download and checksum the archive**

Write `scripts/fetch_trace.sh` to pull the Drive file (the confirm-token dance
is needed — it is over Drive's virus-scan threshold), unzip to `data/trace/`,
and print `sha256sum` of the zip. Add `data/` to `.gitignore`.

Expected: eight task directories plus Lima, each with three JSON files.
**If the link is dead, STOP and report** — do not silently substitute the
scattered third-party HF re-uploads of the constituent tasks. They exist
(`yfhe/C-STANCE-A`, `lytang/MeetingBank-transcript`, `HHazard/numglue`, …) but
with inconsistent schemas and unverified provenance, which would break the
"one format, one loader" property this task exists to establish.

- [ ] **Step 2: Record the checksum in LAB-NOTES**

The archive is now the provenance root of every phase-1/2 result. Write the
sha256, the date fetched, and the task inventory (name → train/eval/test counts)
into `docs/LAB-NOTES.md`. Copy the zip to a second location on the box so a
dead link later is an inconvenience, not a lost phase.

- [ ] **Step 3: Write `src/flab/trace.py`**

One loader, mirroring `data.py`'s shape so both feel like the same codebase:

```python
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

def format_example(ex: dict) -> str: ...      # prompt/answer -> the <|user|>/<|assistant|> tags data.py already uses
def load_task(name, n_train=None, n_eval=200, seed=0) -> DatasetDict: ...
```

Reuse `data.py`'s `TAGS` rather than inventing a second chat format — the
conversion corpus and the task corpus must not differ in formatting, or the
format change becomes a confound in the forgetting measurement.

**Truncate the prompt from the left; never truncate the answer.** Answer-token
NLL *is* the measurement, so an example whose answer got cut by the window is
not a noisy data point, it is a corrupted one. Left-truncation also keeps the
tokens immediately preceding the completion, which is the context that matters
for Py150's 13% long tail. Assert in the loader that every formatted example
retains its full answer, and count how many prompts were truncated so the
number lands in LAB-NOTES rather than staying invisible.

- [ ] **Step 4: Test and commit**

`tests/test_trace.py`: every task in `TASKS` loads; splits are disjoint;
formatting round-trips; `n_eval` is honoured. Tests must skip cleanly (not
fail) when `data/trace/` is absent, so a fresh clone still passes `pytest`.

```bash
uv run pytest -q
git add scripts/fetch_trace.sh src/flab/trace.py tests/test_trace.py .gitignore docs/LAB-NOTES.md
git commit -m "feat: TRACE data loader; vendor and checksum the benchmark archive"
```

---

### Task 2: Run config and run state

**Files:**
- Create: `src/flab/runconfig.py`
- Create: `src/flab/runstate.py`
- Create: `configs/dev-3stage.yaml`
- Create: `tests/test_runstate.py`

**Interfaces:**
- Produces: `RunConfig.load(path)` → validated config + stable content hash;
  `RunState` → resume-at-stage-k across process restarts.
- Consumes: nothing GPU-side. **This whole task is CPU-only and testable
  without the card** — write it while the A/B or any other run occupies the GPU.

- [ ] **Step 1: Config schema**

```yaml
run_name: dev-3stage
model: HuggingFaceTB/SmolLM2-360M
seed: 0
mode: lora                  # lora | full   <- 1c needs `full`; ternary has no PEFT escape hatch
optim: adamw_torch          # adamw_bnb_8bit for full FT: 6.81 B/param measured
stages:
  - {task: FOMC,      max_steps: 200, learning_rate: 2.0e-4}
  - {task: Py150,     max_steps: 200, learning_rate: 2.0e-4}
  - {task: ScienceQA, max_steps: 200, learning_rate: 2.0e-4}
probe:
  tasks: all                # every task in `stages`, at EVERY boundary - not just seen ones
  n_eval: 200
  max_length: 1024
```

`probe.tasks: all` means every task named in `stages` (3 here → a 3×3 matrix
over 4 boundaries), *not* all nine vendored sets. It also accepts an explicit
list, so probing held-out tasks the run never trains on — pure forward transfer
— stays available without a code change.

**Why these three** — chosen from the measured length profile below, not from
the task names. Classification → code → science QA is about as far apart in
token distribution and output shape as TRACE goes *while staying inside a 360M
English model's competence and inside seq 1024*.

Two candidates were ruled out by measurement rather than taste:

- **MeetingBank is disqualified at seq 1024** — 58% of its examples exceed the
  window. Truncating the majority of a summarization set whose target describes
  the *whole* transcript does not make the task hard, it makes it ill-posed,
  and the model would be learning to confabulate from a fragment.
- **The maximal-shift triple** (C-STANCE Chinese → Py150 → 20Minuten German)
  is tempting but SmolLM2-360M is English-trained and would sit near floor on
  both multilingual sets — reintroducing exactly the no-dynamic-range problem
  that made phase 0's accuracies useless. Forgetting you cannot resolve is not
  a result.

Measured over `LLM-CL-Benchmark_5000`, 2026-08-08 (prompt/answer characters):

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

The chosen trio also spans three answer-length regimes — 1 token, ~8 tokens,
~200 tokens — which is a free stress test of the probe. This is safe *because*
NLL is only ever compared against the same task's own baseline, never across
tasks (spec §9).

`mode` and `optim` exist from day one specifically so phase 1c does not need a
harness rewrite — the ternary recipe is a full fine-tune, and 8-bit Adam is the
single highest-leverage memory choice available to it (spec §4).

- [ ] **Step 2: Config hash + provenance**

`RunConfig` exposes `content_hash` (sha256 of the canonicalised config). On run
start, write `run.json` into the run directory: config hash, full config copy,
`git rev-parse HEAD`, whether the tree was dirty, and the TRACE archive
checksum from task 1. That tuple is the spec's "re-runnable from a commit hash"
deliverable — a result is only as reproducible as its data provenance.

- [ ] **Step 3: Run state**

```json
{"run_name": "...", "config_hash": "...",
 "stages": [{"name": "C-STANCE", "status": "done|running|pending|failed",
             "steps_done": 200, "checkpoint": "...", "probe": "probe-after-0.json"}]}
```

Written atomically (temp file + `os.replace`) after every state transition — a
crash mid-write must not corrupt the file that makes recovery possible.

- [ ] **Step 4: Test and commit**

`tests/test_runstate.py` must cover the failure the design exists for: write
state, simulate a kill, reload, assert the harness resumes at the first
non-`done` stage. Also assert a changed config hash refuses to resume into a
run directory built from a different config — silently resuming a *different*
experiment into an existing run dir is the worst failure this file can have.

```bash
uv run pytest -q
git add src/flab/runconfig.py src/flab/runstate.py configs/ tests/test_runstate.py
git commit -m "feat: declarative run config with provenance hash; resumable run state"
```

---

### Task 3: The likelihood probe

This is the instrument. Phase 0 established that at this scale benchmark
accuracy resolves nothing while held-out loss resolves clearly, so the harness's
primary output is a **loss matrix**, not an accuracy table.

**Files:**
- Create: `src/flab/probes.py`
- Modify: `docs/LAB-NOTES.md` (measured probe cost)

**Interfaces:**
- Produces: `probe_all(model, tokenizer, tasks, cfg) -> dict` — per-task mean
  held-out NLL and token accuracy, plus n_tokens and a `warning` field.

- [ ] **Step 1: Write the probe**

Forward-only, no generation, `torch.no_grad()`, batched, bf16 autocast. For
each task's held-out split: mean per-token NLL over **answer tokens only**
(prompt tokens are identical across stages and would dilute the signal), token
accuracy, and token count.

**Never average NLL across tasks, and always report `n_tokens`.** The chosen
trio spans 1 token of answer (FOMC's single letter) to ~200 (ScienceQA), so a
cross-task mean would be dominated by whichever task is wordiest and would move
when nothing had changed. Each task is compared only against its own
pre-stage-0 baseline. FOMC's one-token answer is not a defect — it makes that
column a clean classification log-loss, which is strictly more sensitive than
the accuracy phase 0 showed to be useless.

Follow `mem_probe.py`'s habit and emit an explicit **`warning` field** that is
non-null when the probe could not actually measure what it claims — the first
memory probe returned a plausible-looking `0.0` because a callback was never
registered, and the warning field is what caught it. A probe that can't say
"I did not measure this" will eventually put a fabricated number in LAB-NOTES.

- [ ] **Step 2: Probe every task at every boundary**

After stage *k*, probe **all N tasks**, not only those trained so far. The
resulting N×N matrix yields forgetting, backward transfer and forward transfer
without any further runs; probing only seen tasks throws away the forward-transfer
half for no saving worth having.

- [ ] **Step 3: Merge adapters before any generative eval**

If a generative eval is ever added at a boundary, it operates on a
`merge_and_unload()` copy. Measured: an unmerged adapter costs **1.89×** on
generative decode. Assert this in code, not in a comment — the harness should
refuse to run a generative eval on a live PEFT model.

- [ ] **Step 4: Measure the probe's real cost, and write it down**

**Computed estimate, to be replaced by measurement:** 8 tasks × 200 held-out
sequences at seq 1024, forward-only, is order **4 min per boundary** and
~30 min across an 8-stage run. If that holds, the full forgetting instrument
costs less than *one third* of a single IFEval run (1 h 50 m). That is the
phase-0 lesson turned into a design.

Run the probe standalone against the base model, time it, and put the measured
number in LAB-NOTES beside the estimate. **Do not carry the estimate forward as
if it were a result** (CLAUDE.md hard rule 3).

```bash
git add src/flab/probes.py docs/LAB-NOTES.md
git commit -m "feat: held-out NLL probe; measured boundary-eval cost"
```

---

### Task 4: The sequential harness

**Files:**
- Create: `src/flab/sequential.py`
- Create: `tests/test_sequential.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `uv run python -m flab.sequential --config configs/dev-3stage.yaml`

- [ ] **Step 1: The stage loop**

```
load config -> open/create run dir -> load or init run state
probe once before stage 0            <- the baseline every later number is relative to
for each pending stage:
    train (TRL, resume_from_checkpoint if a checkpoint exists)
    mark stage done, save checkpoint
    probe all tasks -> probe-after-<k>.json
write completion marker
```

The pre-stage-0 probe matters for a reason the spec calls out in §9: the ternary
twin starts weaker than its float twin, so forgetting must be measured relative
to **each model's own post-conversion baseline**, never as a cross-model
absolute. The harness should make the right comparison the easy one.

- [ ] **Step 2: Mid-stage checkpoints, never speculatively evaluated**

`save_steps` writes weights to disk during a stage; the probe fires only at
stage boundaries (spec §6 1a). Mid-stage checkpoints are crash insurance, not
data points.

- [ ] **Step 3: LoRA and full modes through one code path**

`mode: lora` attaches a `LoraConfig`; `mode: full` passes no `peft_config` and
uses `optim` from the config. Both must reach the same probe code, or 1c will
discover a fork in the instrument at the worst possible moment.

- [ ] **Step 4: Test on CPU with a tiny model**

`tests/test_sequential.py`: two stages, 2 steps each, a tiny HF test model, on
CPU. Assert stage ordering, that state advances, that a mid-run kill resumes at
the right stage, and that probe files are written per boundary. Fast enough for
`uv run pytest` to stay a pre-commit habit.

```bash
uv run pytest -q
git add src/flab/sequential.py tests/test_sequential.py
git commit -m "feat: sequential fine-tuning harness with boundary probes"
```

---

### Task 5: Supervisor — crash-resume and auto-retry

Spec §6 1a asks for this "from day one": at 5–10 h/week of human attention, a
3 a.m. OOM must cost minutes, not a calendar day.

**Files:**
- Create: `scripts/run_sequential.sh`

- [ ] **Step 1: Retry loop**

Relaunch the harness on non-zero exit, up to `MAX_RETRIES` (default 3), with the
resume path taken automatically because run state is on disk. Log each attempt
with a timestamp.

- [ ] **Step 2: Distinguish retryable from fatal**

An OOM is **not** retryable at the same batch size — retrying it three times
just wastes an hour and heat-soaks the chassis for whatever runs next. Grep the
failure log: on OOM, stop and report loudly. On anything else, retry.

- [ ] **Step 3: Completion marker**

Write the marker at the end of the chain, and follow the existing convention:
a marker means *finished*, not *worked*. Include the last exit code in it so a
watcher can tell the difference without parsing logs.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sequential.sh
git commit -m "feat: supervising run script with bounded auto-retry"
```

---

### Task 6: Shakedown run + the deliverable

- [ ] **Step 1: Three-stage dev run**

`tmux new -As seq` → `scripts/run_sequential.sh configs/dev-3stage.yaml`.
3 stages × 200 steps, LoRA, seed 0.

**Budget: ~2–3 GPU-h derated** (3 × 200 steps at the measured heat-soaked
4.85→9.43 s/step, plus 4 boundary probes). Well inside the 40 GPU-h card cap.

**No design card required** (Arley, 2026-08-08). This is harness validation, not
a result-bearing run — the same category as the thermal, memory and clock-cap
probes. The gate still stands for everything downstream of it: the first run
that produces a *number we would report* needs a card.

- [ ] **Step 2: Prove the resume path on the real thing**

Kill the tmux session mid-stage-2 and restart it. Assert it resumes inside
stage 2 rather than restarting stage 0. A resume path that has only ever been
tested on CPU with a toy model is not a resume path you can trust at 3 a.m.

- [ ] **Step 3: Read the loss matrix**

Assemble the 4 probe files into the N×N matrix and plot it. **Expect little or
no forgetting** — three short LoRA stages on a 360M model is a small
intervention, and phase 0 showed how insensitive this scale is. The deliverable
is that the instrument *reads*, not that it finds something. A clean null here
is a working harness, not a failed experiment.

- [ ] **Step 4: LAB-NOTES + commit**

Record: measured probe cost vs the task-3 estimate, wall-clock per stage,
peak reserved VRAM, whether resume worked on the real run, and the matrix.

```bash
git add docs/LAB-NOTES.md configs/
git commit -m "docs: phase-1a shakedown — harness reads, resume verified"
```

Phase 1a is complete when: a config file plus a commit hash reproduce a run, a
kill -9 costs minutes, and the loss matrix reads. Next: **1b calibration gate**
(spec §6) — replicate 2606.27634's trends and the synthetic control before any
result-bearing run. That gets its own plan.

---

## Decisions taken (2026-08-08)

- **Three tasks, not eight.** Vendor all nine; run three. Revisit after 1b.
- **No design card for the shakedown.** Engineering validation, like the
  thermal/memory/clock probes. The gate still applies to the first run whose
  numbers we would report.

## Open — does not block 1a

1. **Disk policy for 1c/1d.** LoRA adapters are ~35 MB, so phase 1a is free.
   Phase 1d wants *full latent checkpoints throughout*, which at 360M is
   ~1.4 GB per checkpoint in bf16 (more with optimizer state). At every-100-steps
   across an 8-stage run that is comfortably into the hundreds of GB. The 1 TB
   NVMe can take it, but the retention policy should be a decision, not an
   accident. Flagging now; it does not block 1a.
