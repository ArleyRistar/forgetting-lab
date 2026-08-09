"""BitLinear — ternary (1.58-bit) linear layers via continued QAT (spec §6 1c).

Reproduces the recipe from the HF Llama3→1.58bit work. Quantisers are module-level
functions so they can be tested against hand-computed values: testing numerics
through a model is how the KL-direction bug survived in 1b, because a small
model's outputs hide almost everything.

Two properties gate everything downstream, and both are tested:

* **λ=0 is bit-identical to the original `nn.Linear`.** Phase 1c's premise is
  that at the moment of conversion "the float weights *are* the initial latent
  weights". If λ=0 is not exactly the float layer, that premise is false and the
  latent trajectory phase 1d measures is not the float model's.
* **λ=1 actually ternarises.** Effective weights take at most 3 distinct values.
  Without this we could train a float model for 30 GPU-h and report it as
  ternary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear submodules replaced in a Llama-style block. Embeddings, lm_head and
# norms are deliberately excluded — the recipe quantises attention and FFN only.
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Ternarise to {-s, 0, +s} with an absmean scale.

    Note the scale is per *tensor*, recomputed every forward pass, so the
    threshold between 0 and ±1 moves as the latent weights move. Phase 1d must
    therefore define a flip by **effective-value change, not latent-value
    change** (spec §6 1d) — a weight can flip without moving.
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax quantisation to int8 range."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127) / scale


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMS norm applied inside BitLinear before quantising.

    The recipe calls normalisation before activation quantisation **essential**,
    and skipping it is what broke the first 135M shakedown: loss climbed
    smoothly with lambda to 11.75, above the ~10.8 of a uniform guess.

    Per-token absmax quantisation divides by the largest activation in each
    token, so a layer whose inputs have wide dynamic range wastes most of the
    int8 grid. A transformer block's own norms do not cover this: they normalise
    the *block* input, while `o_proj` sees raw attention output and `down_proj`
    sees the raw SwiGLU product. Those two are exactly the layers that break.

    Parameter-free deliberately — a learnable norm would add randomly
    initialised parameters, breaking both the unchanged-parameter-count
    invariant and the premise that the float weights are the initial latents.
    """
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x32.to(dtype)


def ste(x: torch.Tensor, quantised: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Straight-through estimator with warmup.

    Forward: x + λ(q(x) − x), so λ=0 returns x exactly and λ=1 returns q(x).
    Backward: the correction is detached, so gradients pass through as if the
    quantiser were the identity — which is the only reason latent weights can
    be trained at all.
    """
    return x + lambda_ * (quantised - x).detach()


class BitLinear(nn.Linear):
    """`nn.Linear` whose weights ternarise and activations quantise in forward.

    Subclasses `nn.Linear` rather than wrapping it so the latent weights stay
    the *same parameter objects* — converting must not rebuild the model, or
    the trajectory stops being the float model's.
    """

    def __init__(self, *args, lambda_: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Not a buffer: it is a schedule value, not model state, and saving it
        # into checkpoints would make resume depend on where warmup happened to be.
        self.lambda_ = lambda_

    @classmethod
    def from_linear(cls, linear: nn.Linear, lambda_: float = 0.0) -> BitLinear:
        out = cls(linear.in_features, linear.out_features,
                  bias=linear.bias is not None, lambda_=lambda_,
                  device=linear.weight.device, dtype=linear.weight.dtype)
        # Share the tensors rather than copy: same parameter objects, same
        # trajectory.
        out.weight = linear.weight
        if linear.bias is not None:
            out.bias = linear.bias
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lambda_ == 0.0:
            # Exact no-op path. Going through the STE with λ=0 would be
            # mathematically identical but not bitwise, and the premise of the
            # conversion is bitwise equality at the moment of conversion.
            return F.linear(x, self.weight, self.bias)
        # Normalise, then quantise — with the norm interpolated by lambda too,
        # so lambda=0 stays exactly the float layer. The recipe applies the norm
        # unconditionally; doing that here would mean the conversion starts from
        # a function that is not the float model, and phase 1c's premise is that
        # it is.
        xn = x + self.lambda_ * (rms_norm(x) - x)
        x = ste(xn, activation_quant(xn), self.lambda_)
        w = ste(self.weight, weight_quant(self.weight), self.lambda_)
        return F.linear(x, w, self.bias)


def convert(model: nn.Module, lambda_: float = 0.0) -> tuple[nn.Module, int]:
    """Replace attention/FFN linears in place. Returns (model, n_replaced)."""
    n = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, BitLinear) \
                    and name.endswith(TARGET_SUFFIXES):
                setattr(module, name, BitLinear.from_linear(child, lambda_))
                n += 1
    return model, n


def set_lambda(model: nn.Module, lambda_: float) -> int:
    """Push the warmup value into every BitLinear. Returns how many were set."""
    n = 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.lambda_ = lambda_
            n += 1
    return n


def warmup_lambda(step: int, total: int = 1000) -> float:
    """`min(step/total, 1)` — the schedule the writeup found worked best.

    Kept at 1000 steps regardless of budget: it exists to stop the model
    collapsing when quantisation switches on, and that risk does not shrink
    with fewer total steps.
    """
    return min(step / total, 1.0) if total > 0 else 1.0
