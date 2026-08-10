"""Flip/distance predictors for each phase-2 run (A-end -> B-end).

Runs over `outputs/phase2/results.json` and, for every row, computes the
predictors H1 compares against forgetting — on identical layers and identical
checkpoint pairs, which is the only way the comparison is fair:

  flip fraction, four-way partitioned (weight-only / scale-only / redundant /
  joint), per-layer flip **concentration**, L2, cosine, and flips/L2.

Concentration is the term most likely to decouple flips from L2: L2 is a global
magnitude, and "which layers moved" is not. Phase 1d found flips = 0.0002*L2 to
within 15% under diffuse drift, so if task shift produces structured motion, it
should show up here rather than in the flip count.

`--prune` deletes each run's checkpoints once its metrics are written. Disk is
the binding constraint at ~5.7 GB per run, and the metrics are what we keep.

Usage:
  uv run scripts/phase2_flips.py
  uv run scripts/phase2_flips.py --prune
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from flab import flips
from flab.bitlinear import TARGET_SUFFIXES

ROOT = Path("outputs/phase2")


def is_target(name: str) -> bool:
    return name.endswith(".weight") and any(
        name[: -len(".weight")].endswith(s) for s in TARGET_SUFFIXES)


def gini(xs: list[float]) -> float:
    """Concentration of flips across layers. 0 = every layer flips equally,
    1 = all flips in one layer."""
    v = sorted(x for x in xs if x >= 0)
    n = len(v)
    if n == 0 or sum(v) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * sum(v)) - (n + 1) / n


def predictors(a_ckpt: Path, b_ckpt: Path) -> dict:
    fa = safe_open(str(a_ckpt / "model.safetensors"), framework="pt")
    fb = safe_open(str(b_ckpt / "model.safetensors"), framework="pt")
    names = sorted(n for n in fa.keys() if is_target(n))

    total = flips.FlipStats()
    l2_sq = cos_w = n_w = 0.0
    per_layer_rate: list[float] = []
    for n in names:
        wa, wb = fa.get_tensor(n).float(), fb.get_tensor(n).float()
        st = flips.flip_partition(wa, wb)
        st.check()
        total = total + st
        per_layer_rate.append(st.flip_fraction)
        d = flips.layer_delta(wa, wb)
        l2_sq += d.l2 ** 2
        cos_w += d.cosine * d.n
        n_w += d.n

    l2 = l2_sq ** 0.5
    return {
        "n_tensors": len(names), "n_weights": total.n,
        "flip_fraction": total.flip_fraction,
        "weight_only": total.weight_only, "scale_only": total.scale_only,
        "redundant": total.redundant, "joint": total.joint,
        "scale_share": (total.scale_only / total.flipped) if total.flipped else None,
        "l2": l2, "cosine": cos_w / n_w if n_w else None,
        "flips_per_unit_l2": (total.flip_fraction / l2) if l2 > 0 else None,
        "flip_concentration_gini": gini(per_layer_rate),
        "per_layer_max_rate": max(per_layer_rate) if per_layer_rate else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prune", action="store_true",
                   help="delete each run's checkpoints once its metrics are saved")
    p.add_argument("--out", default=str(ROOT / "predictors.json"))
    a = p.parse_args()

    rows = json.loads((ROOT / "results.json").read_text())
    done = {}
    out_path = Path(a.out)
    if out_path.is_file():
        done = {r["key"]: r for r in json.loads(out_path.read_text())}

    result = list(done.values())
    for row in rows:
        key = f"{row['arm']}-{row['pair']}-s{row['seed']}-{row['condition']}-{row['b_steps']}"
        if key in done:
            continue
        a_ck, b_ck = Path(row["a_ckpt"]), Path(row["b_ckpt"])
        if not (a_ck / "model.safetensors").is_file() or \
           not (b_ck / "model.safetensors").is_file():
            print(f"  {key}: checkpoints missing, skipping", flush=True)
            continue
        pred = predictors(a_ck, b_ck)
        rec = {"key": key, **{k: row[k] for k in
               ("arm", "pair", "seed", "condition", "b_steps", "forgetting")}, **pred}
        result.append(rec)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  {key:<44} flips {pred['flip_fraction']:.4%}  L2 {pred['l2']:.3f}  "
              f"f/L2 {pred['flips_per_unit_l2']:.6f}  gini {pred['flip_concentration_gini']:.3f}",
              flush=True)
        if a.prune:
            shutil.rmtree(b_ck, ignore_errors=True)

    if a.prune:
        # A-checkpoints are shared by every branch, so they go only at the end.
        for row in rows:
            shutil.rmtree(Path(row["a_ckpt"]), ignore_errors=True)
        print("pruned checkpoints; metrics retained", flush=True)
    print(f"\n{len(result)} rows -> {out_path}")


if __name__ == "__main__":
    main()
