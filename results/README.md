# Recorded results

The JSON behind every number in `docs/blog/` and `docs/LAB-NOTES.md`. Copied out
of `outputs/` (which is gitignored — it also holds ~250 GB of checkpoints) so the
figures regenerate and the claims can be checked from a fresh clone.

| file | what it holds |
| --- | --- |
| `convert-conversion-gap.json` | held-out loss for base / float twin / ternary twin, both corpora |
| `convert-zero-fraction.json` | zero occupancy of the converted twin, per projection |
| `convert-flips-{ternary,float}.json` | flip trajectory across the conversion checkpoints |
| `null-capability-gate.json` | the item-20 gate: discrimination vs shuffled answers |
| `null-flips-constlr.json`, `null-lag-curve.json` | per-step flip floor and its accumulation curve |
| `phase2-analysis.json`, `phase2-predictors.json` | the 72-run sweep and its predictors |
| `phase2b-analysis-v2.json` | corrected phase-2b contrasts, incl. the scale-free rank statistic |
| `ternary-batch-stability.json` | batch-invariance drift, the quantiser ablation, and the row-local scale check |

Regenerate the post's figures with `uv run python scripts/blog_figures.py`.
