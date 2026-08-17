# ternary-instruments

Instruments for measuring what happens to a small language model when you convert
it to ternary (1.58-bit) weights — and a record of how often those instruments
turned out to be measuring something else. Run on a single RTX 3070 Ti Laptop
(8 GB). Arley Ristar directs; Claude implements.

It began as **forgetting-lab**, asking whether ternary models forget differently.
That programme is closed: the primary hypothesis was refuted and the secondary
one retracted. What survived is about measurement, which is why the name changed.
The write-up is at
[`docs/blog/01-ternary-conversion.md`](docs/blog/01-ternary-conversion.md).

The Python package is still `flab`, from the original name. Renaming it would
touch every import, test and lab-notes reference for no reader benefit.

**Looking for something usable without reading the notebook?**
[`FINDINGS.md`](FINDINGS.md) is the dense version: every transferable result with
its number, the conditions it holds under, and where to verify it — including the
dead ends and the one retraction, so nobody rebuilds them.

## What is here that might be useful to you

- **A ternary conversion recipe with a data-matched float twin** — the same base
  model trained on the *same tokens* in float, as a control. `src/flab/convert.py`.
- **The bf16 latent-weight trap**: at lr 1e-4, bf16 latents round ~85% of Adam
  updates to zero and conversion silently fails. See the blog post.
- **A weight-state flip instrument** with a four-way causal partition and
  planted-effect tests. `src/flab/flips.py`, `tests/test_flips.py`.
- **A guard against likelihood-only capability claims**, written after we
  retracted one. `src/flab/claims.py`.
- **Every number, including the retracted ones**: `docs/LAB-NOTES.md` is the
  durable record, newest at the bottom. It is long and it does not tidy up its
  own mistakes.

## Running it

Python is pinned to 3.12 (`.python-version`); 3.14 lacks the dev headers the
low-bit stack needs.

```bash
uv sync
uv run pytest                    # 149 pass; 44 skip until the TRACE data is fetched
scripts/fetch_trace.sh           # optional, for the benchmark tasks
```

Reproduce the matched pair (~13 GPU-h on one 8 GB card):

```bash
uv run python -m flab.convert --model HuggingFaceTB/SmolLM2-360M --mode ternary \
  --max-steps 4000 --lr 1e-4 --optim adamw_bnb_8bit

uv run python -m flab.convert --model HuggingFaceTB/SmolLM2-360M --mode float \
  --max-steps 4000 --lr 1e-4 --optim adamw_bnb_8bit \
  --batch-size 2 --grad-accum 8 --expect-tokens-per-step 16384
```

The float arm needs the smaller micro-batch: under autocast it passes leaf
parameters to `F.linear`, so a bf16 copy of every weight is cached, while the
ternary path passes a computed STE tensor and bypasses that. Same tokens/step
either way.

## Layout

| path | what |
| --- | --- |
| `src/flab/` | the package — conversion, BitLinear, flips, probes, harness, guards |
| `scripts/` | measurement tools, deliberately not part of the package |
| `results/` | the JSON behind every published number (`outputs/` is gitignored) |
| `docs/LAB-NOTES.md` | the durable record of every measurement |
| `docs/superpowers/specs/` | the design spec |
| `docs/superpowers/plans/` | one design card per experiment |
| `docs/blog/` | write-ups |

## One caveat on every number here

65.5M conversion tokens is roughly 150× less than the reference recipe. Read the
results as "what happens at a hobby budget", not as what ternary conversion can
do.
