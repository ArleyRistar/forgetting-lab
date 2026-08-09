"""One load path for converted twins, and a guard that they really are ternary.

**Why this module exists.** Ternary checkpoints hold *latent* weights —
ternarisation happens in the forward pass, so the saved tensors are the
full-precision latents (`convert.py`, "Checkpoints hold latent weights"). A
caller that does `AutoModelForCausalLM.from_pretrained(ternary_path)` therefore
gets a **float model**, and every number it produces is a float number wearing a
ternary label. Nothing crashes. Nothing looks wrong.

That is this project's recurring failure shape — the instrument silently
measuring a different object than its name claims. It has now happened four
times (KL direction, turn terminator, teacher forcing, and a "held-out" split
that was not held out), and each earlier instance was caught only *after* a
result had been recorded or retracted. Before this module, `conversion_gap.py`
was the only caller that re-applied BitLinear; `sequential.py`, `probes.py`,
`clmetrics.py` and `eval.sh` did not, so phase 2 would have fine-tuned and
scored a float model while calling it ternary.

So: one function everything goes through, and an assertion on the *effective*
weights rather than a promise in a docstring. `assert_ternary` checks what the
forward pass would actually compute, which is the only claim worth making.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from flab import bitlinear as bl

# A ternary run writes this next to its output; `save_model` puts the weights in
# a `final/` (or `checkpoint-N/`) subdirectory, so look upward as well.
CONVERT_META = "convert.json"


def find_convert_meta(path: str | Path) -> dict | None:
    """Return the `convert.json` governing this checkpoint, or None.

    Searched in the checkpoint directory and its parent, because
    `outputs/convert/ternary-360m/convert.json` governs
    `outputs/convert/ternary-360m/final/`.
    """
    p = Path(path)
    for candidate in (p / CONVERT_META, p.parent / CONVERT_META):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except json.JSONDecodeError:
                return None
    return None


def is_ternary_checkpoint(path: str | Path) -> bool:
    """True when this checkpoint's weights are ternary latents.

    Deliberately keyed on the run's own recorded mode rather than on the
    directory name: a path called `ternary-360m` proves nothing, and a run that
    never finished warmup is not a ternary model whatever it is called.
    """
    meta = find_convert_meta(path)
    if meta is None:
        return False
    if meta.get("config", {}).get("mode") != "ternary":
        return False
    if not meta.get("warmup_completed", False):
        raise ValueError(
            f"{path} is a ternary run whose warmup never completed "
            f"(final_lambda={meta.get('final_lambda')}). Its weights are not "
            "ternary and must not be reported as such")
    return True


@torch.no_grad()
def assert_ternary(model: nn.Module, max_levels: int = 3) -> int:
    """Assert every BitLinear's *effective* weights take <= 3 distinct values.

    Checks what the forward pass computes, not what the module is called. A
    model that silently failed to convert has BitLinears whose effective weights
    are continuous, and this is the only cheap way to notice.

    Returns the number of layers checked, so a caller can assert it is nonzero —
    a model with zero BitLinears trivially satisfies "all of them are ternary".
    """
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, bl.BitLinear):
            continue
        if m.lambda_ != 1.0:
            raise AssertionError(
                f"{name} has lambda={m.lambda_}, so its forward pass is not "
                "fully ternary")
        levels = torch.unique(bl.weight_quant(m.weight.float())).numel()
        if levels > max_levels:
            raise AssertionError(
                f"{name} effective weights take {levels} distinct values, "
                f"expected <= {max_levels} — this is not a ternary layer")
        n += 1
    return n


def load_converted(path: str | Path, *, dtype=torch.float32,
                   device: str | None = None, force_ternary: bool | None = None):
    """Load a twin, re-applying BitLinear at lambda=1 when it is ternary.

    This is the *only* supported way to load a phase-1c twin. `force_ternary`
    overrides the `convert.json` detection for checkpoints written before the
    metadata existed; leave it None in normal use.

    Returns `(model, n_bitlinear)`. `n_bitlinear == 0` means a float model — so
    a caller expecting the ternary arm should assert it is nonzero rather than
    trusting the path it passed in.
    """
    ternary = is_ternary_checkpoint(path) if force_ternary is None else force_ternary
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    n = 0
    if ternary:
        model, n = bl.convert(model, lambda_=1.0)
        checked = assert_ternary(model)
        if checked != n:
            raise AssertionError(
                f"converted {n} layers but only verified {checked} as ternary")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), n
