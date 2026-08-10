"""Weight-state flip trajectory over saved latent checkpoints (phase 1d, task 2).

Reads tensors lazily one at a time via `safe_open` rather than instantiating
models: the analysis is per-tensor anyway (the absmean scale is per-tensor), and
holding five 360M checkpoints in memory to compute a per-tensor statistic would
be wasteful on an 8 GB box.

**The float twin's flips are counterfactual and labelled as such.** Its weights
are not ternary, but applying the same state function to them answers a question
H1 needs: how much would a float model's would-be ternary states move under the
same training? Without it, "ternary training causes flips" is not separable from
"any training moves weights across a hypothetical threshold".

Usage:
  uv run scripts/flip_report.py --arm ternary
  uv run scripts/flip_report.py --arm float
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from flab import flips
from flab.bitlinear import TARGET_SUFFIXES

BASE = "HuggingFaceTB/SmolLM2-360M"


def is_target(name: str) -> bool:
    return name.endswith(".weight") and any(
        name[: -len(".weight")].endswith(s) for s in TARGET_SUFFIXES)


def proj_kind(name: str) -> str:
    return name[: -len(".weight")].rsplit(".", 1)[-1]


def base_state_dict() -> dict[str, torch.Tensor]:
    """Step 0 — the base model's weights ARE the initial latents (spec §6 1c)."""
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    return {k: v.detach() for k, v in m.state_dict().items() if is_target(k)}


class Source:
    """Uniform lazy accessor over either a safetensors checkpoint or a dict."""

    def __init__(self, label: str, path: Path | None, mem: dict | None = None):
        self.label, self.path, self.mem = label, path, mem
        self._f = None
        if path is not None:
            self._f = safe_open(str(path / "model.safetensors"), framework="pt")

    def keys(self):
        return list(self.mem) if self.mem is not None else \
            [k for k in self._f.keys() if is_target(k)]

    def get(self, name: str) -> torch.Tensor:
        return self.mem[name] if self.mem is not None else self._f.get_tensor(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("ternary", "float"), default="ternary")
    p.add_argument("--root", default=None)
    p.add_argument("--steps", default="1000,2000,3000,4000")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.root or f"outputs/convert/{a.arm}-360m")
    steps = [int(s) for s in a.steps.split(",")]

    print("loading step 0 (base model = initial latents)", flush=True)
    sources = [Source("step0", None, base_state_dict())]
    for s in steps:
        d = root / f"checkpoint-{s}"
        if not d.is_dir():
            print(f"  {d} missing, skipping", flush=True)
            continue
        sources.append(Source(f"step{s}", d))
    labels = [s.label for s in sources]
    print(f"arm={a.arm} checkpoints: {labels}", flush=True)

    names = sorted(n for n in sources[-1].keys() if is_target(n))
    print(f"{len(names)} target tensors", flush=True)

    # interval -> aggregate stats, and per projection kind
    agg: dict[str, flips.FlipStats] = defaultdict(flips.FlipStats)
    per_kind: dict[tuple[str, str], flips.FlipStats] = defaultdict(flips.FlipStats)
    agg_cur: dict[str, flips.FlipStats] = defaultdict(flips.FlipStats)
    dist: dict[str, dict] = defaultdict(lambda: {"l2": 0.0, "cos": 0.0, "n": 0})
    persist: dict[int, list[int]] = {k: [0, 0] for k in (1, 2)}
    hist: dict[str, torch.Tensor] = {}

    for i, name in enumerate(names):
        tensors = [s.get(name).float() for s in sources]
        states = [flips.state(t, flips.tensor_scale(t)) for t in tensors]

        for j in range(1, len(tensors)):
            iv = f"{labels[j-1]}->{labels[j]}"
            st = flips.flip_partition(tensors[j - 1], tensors[j], reference="prev")
            st.check()
            agg[iv] = agg[iv] + st
            per_kind[(iv, proj_kind(name))] = per_kind[(iv, proj_kind(name))] + st
            agg_cur[iv] = agg_cur[iv] + flips.flip_partition(
                tensors[j - 1], tensors[j], reference="cur")

            d = flips.layer_delta(tensors[j - 1], tensors[j])
            dist[iv]["l2"] += d.l2 ** 2          # sum of squares, rooted below
            dist[iv]["cos"] += d.cosine * d.n
            dist[iv]["n"] += d.n

        for k in persist:
            if len(states) > k + 1:
                h, c = flips.persistence(states, k)
                persist[k][0] += h
                persist[k][1] += c

        for j, t in enumerate(tensors):
            hist[labels[j]] = hist.get(labels[j], 0) + flips.distance_to_threshold(t)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(names)} tensors", flush=True)

    out = {
        "arm": a.arm,
        "counterfactual": a.arm == "float",
        "checkpoints": labels,
        "n_tensors": len(names),
        "intervals": {},
        "per_projection": {},
        "persistence": {str(k): {"held": v[0], "considered": v[1],
                                 "fraction": (v[0] / v[1]) if v[1] else None}
                         for k, v in persist.items()},
        "distance_histogram": {k: v.tolist() for k, v in hist.items()},
        "histogram_bins": {"n": 40, "lo": 0.0, "hi": 2.0,
                           "boundary_at": 0.5, "clamp_edge_at": 1.5},
    }
    for iv, st in agg.items():
        st.check()
        d = dist[iv]
        out["intervals"][iv] = {
            **st.as_dict(),
            "reference_cur_flip_fraction": agg_cur[iv].flip_fraction,
            "l2": d["l2"] ** 0.5,
            "cosine_weighted": d["cos"] / d["n"] if d["n"] else None,
            # The comparison H1 lives or dies on: if flips are just a rescaled
            # L2, they cannot predict forgetting *better* than L2 does.
            "flips_per_unit_l2": (st.flip_fraction / (d["l2"] ** 0.5)
                                  if d["l2"] > 0 else None),
        }
    for (iv, kind), st in per_kind.items():
        out["per_projection"].setdefault(iv, {})[kind] = {
            "flip_fraction": st.flip_fraction,
            "weight_only": st.weight_only, "scale_only": st.scale_only,
            "redundant": st.redundant, "joint": st.joint,
            "zero_fraction_cur": st.zero_cur / st.n if st.n else None,
        }

    dest = Path(a.out or f"outputs/convert/flips-{a.arm}.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    for iv, v in out["intervals"].items():
        print(f"  {iv:<18} flips {v['flip_fraction']:.4%}  "
              f"weight-only {v['weight_only']:>9,}  scale-only {v['scale_only']:>9,}  "
              f"redundant {v['redundant']:>9,}  joint {v['joint']:>7,}")


if __name__ == "__main__":
    main()
