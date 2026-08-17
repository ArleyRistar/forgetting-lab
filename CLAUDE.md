# CLAUDE.md — forgetting-lab

Hobby ML research: **catastrophic forgetting in ternary (1.58-bit) LLMs**.
Arley directs, Claude implements. Read `docs/superpowers/specs/2026-08-05-forgetting-lab-design.md`
(the spec) before any substantive change — it is the authority on what this
project is for and what is deliberately out of scope.

**If you are an agent arriving here cold:** read [`FINDINGS.md`](FINDINGS.md)
first. It is the transferable results in dense form — numbers, conditions,
verification paths, and the dead ends — and it will save you reading the notebook
to find out what is already known.

## The two documents that matter

| file | holds | when to read |
| --- | --- | --- |
| `docs/superpowers/specs/…-design.md` | design, phases, hardware envelope, killed alternatives | before proposing anything |
| `docs/LAB-NOTES.md` | every measured fact about the lab box, newest at the bottom | before estimating, debugging, or budgeting |

**Write findings into LAB-NOTES, not into memory or a chat message.** Three
copies of this repo exist (Zenbook, lab box, GitHub) and the notes are the only
channel that reaches all of them and survives a new session. If a run produced a
number, the number belongs in the notes with how it was measured.

## Hard rules

1. **No experiment runs without a design card Arley approved** (spec §3) —
   hypothesis, method, metric, seeds, estimated GPU-hours. No card may exceed
   ~40 thermal-derated GPU-h. This gate is about spending compute on
   result-bearing runs; engineering measurements (thermal, memory, throughput)
   have in practice been treated as ordinary work.
2. **Conclusions are Arley's; the blog-post drafting is Claude's** (changed
   2026-08-12 — the original rule reserved all writing for Arley because the
   writing was where his learning happened). Claude now drafts the posts in full;
   Arley edits, approves and decides what is claimed. Claude still never decides
   *what the result is* — every claim must trace to a measured number in
   LAB-NOTES, and anything the notes mark as retracted, superseded or
   inconclusive must be written that way.
3. **Distinguish measured from computed.** Tonight's VRAM table was computed,
   then measured, and the measurement changed two of six numbers. Label which
   is which in the notes; never present arithmetic as a result.
4. **Push freely** — this is Arley's most autonomous project. Commit to `main`,
   no branch needed. Keep all three copies in sync.

## The lab box (`gs66-lab`)

`ssh lab` — MSI GS66, RTX 3070 Ti Laptop 8 GB, headless Fedora 44, cannot
suspend. Repo lives at `~/ternary-instruments`. It has its own repo-scoped GitHub
deploy key, so it can push without a PAT.

**Long runs go in tmux on the box** (`tmux new -As <name>`), never in a
foreground ssh call — an ssh drop or a closed laptop must not kill a run.
Write a completion marker file at the end of a chain so a watcher can tell
"finished" from "still going"; a chain joined with `;` writes its marker even
when a stage failed, so the marker means *finished*, not *worked*.

## Planning numbers — measured, use these

- **Thermal derate ~1.9×.** Step time degrades 4.85 s → 9.43 s as the chassis
  heat-soaks. Budget GPU-hours at 1.9× the cold-start rate.
- **Steady state takes 10+ minutes.** Any thermal or throughput reading before
  ~15 min is misleading. This was misjudged twice during bring-up.
- **Usable VRAM 7.5 GiB of 7.66.** Budget against `peak_reserved`, which runs
  8–13% above `peak_allocated` — that gap is the difference between fitting and
  OOM.
- **Full fine-tune costs** (measured, SmolLM2-360M): bf16+AdamW **8.00**
  B/param, bf16+8-bit Adam **6.81**, fp32 master **16.00** (~18 all-in with
  autocast's weight cache). 8-bit Adam is *not* 6 — bitsandbytes keeps the
  embedding's optimizer state in fp32, which matters most at small scale.
- **Activation memory is logit-dominated** — it scales with
  `vocab × batch × seq`, not model size. The 49k vocab costs the same on a 1.7B
  model as on the 360M.
- **IFEval is expensive**: ~1 h 50 m per 360M model, and **1.89× that again**
  if a LoRA adapter is attached unmerged. Prefer likelihood probes; merge
  adapters before generative evals.

## Environment traps

Full list in LAB-NOTES "Environment quirks"; these cost the most time:

1. **Python is pinned to 3.12** (`.python-version`). Fedora's 3.14 lacks dev
   headers, triton fails to JIT, training dies. The low-bit packages phase 1
   needs have no 3.14 wheels. Do not unpin.
2. **transformers 5.x renamed `torch_dtype` → `dtype`.** Most tutorials online
   still show the old name.
3. **First samples lie.** `nvidia-smi power.draw` returns ~751 W on its first
   read; progress bars claim absurd ETAs on their first estimate. Discard
   sample one, always.
4. **`nvidia-smi -pl` is unsupported on this vBIOS.** Only `-lgc` (clock cap,
   currently 1200 MHz) works. `power.limit` reads `[N/A]`.

## Running things

```bash
uv run pytest                                   # 2 tests, fast
uv run python -m flab.train --max-steps 400     # LoRA SFT smoke test
scripts/eval.sh <tag> [peft-adapter-path]       # lm-eval harness
uv run scripts/mem_probe.py --dtype bfloat16 --optim adamw_torch
```

`src/flab/` is the package (`data.py` formats smol-smoltalk into plain text,
`train.py` is the LoRA SFT entrypoint). `scripts/` holds measurement tools that
are deliberately not part of the package.

## Don't

- Don't estimate GPU-hours from nominal clocks — use the 1.9× derate.
- Don't report a benchmark accuracy delta as a result without checking it
  against its standard error. At this model scale the phase-0 eval moved
  *nothing* beyond ~1.1 SE while held-out loss moved clearly; accuracies are
  not a usable instrument here.
- Don't run a GPU job while another is running — it corrupts both the timing
  and the thermal profile of whatever is already going.
- Don't add dependencies or mutate `.venv` while a run is using it.
