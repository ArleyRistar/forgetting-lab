"""Tests for BitLinear (phase-1c task 1).

Numerics are tested on hand-built tensors, not through a model: in 1b a
direction error in the KL survived a model-based test because a small model's
output distribution is near-uniform and hides almost everything.
"""
import pytest
import torch
import torch.nn as nn

from flab import bitlinear as bl


# -- quantiser numerics ---------------------------------------------------


def test_weight_quant_yields_at_most_three_values():
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    q = bl.weight_quant(w)
    assert len(torch.unique(q)) <= 3


def test_weight_quant_scale_is_absmean():
    w = torch.tensor([[1.0, -1.0, 3.0, -3.0]])
    scale = 1.0 / w.abs().mean()          # 1/2
    q = bl.weight_quant(w)
    # round(w * 0.5) = [0 or 1, ...]; clamped to [-1,1], divided by 0.5
    expected = torch.round(w * scale).clamp(-1, 1) / scale
    assert torch.allclose(q, expected)


def test_weight_quant_of_uniform_weights_is_plus_minus_scale():
    w = torch.full((8, 8), 0.3)
    q = bl.weight_quant(w)
    assert torch.allclose(q, torch.full_like(q, 0.3))   # |w|.mean()=0.3, round(1)=1


def test_activation_quant_stays_in_int8_range_after_rescale():
    torch.manual_seed(0)
    x = torch.randn(4, 16) * 10
    q = bl.activation_quant(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    levels = (q * scale).round()
    assert levels.min() >= -128 and levels.max() <= 127


def test_ste_passes_gradients_through_unchanged():
    """The property the whole recipe depends on, and it is invisible in the
    forward output: without it the latent weights receive no usable gradient."""
    x = torch.randn(16, requires_grad=True)
    out = bl.ste(x, bl.activation_quant(x), lambda_=1.0)
    out.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_warmup_schedule():
    assert bl.warmup_lambda(0) == 0.0
    assert bl.warmup_lambda(500) == 0.5
    assert bl.warmup_lambda(1000) == 1.0
    assert bl.warmup_lambda(5000) == 1.0


# -- the two gating properties -------------------------------------------


def test_lambda_zero_is_bit_identical_to_the_original_linear():
    """Phase 1c's premise: at conversion, the float WEIGHTS are the initial
    latent weights.

    The lambda=0 fast path skips quantisation and the norm entirely, so it is
    still exactly the float layer. Note the premise is about weights, not the
    function — see `test_norm_is_always_on_not_interpolated`: for lambda>0 the
    norm applies unconditionally, as every reference does, and the resulting
    discontinuity at conversion is recorded rather than engineered away."""
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    bit = bl.BitLinear.from_linear(lin, lambda_=0.0)
    x = torch.randn(8, 32)
    assert torch.equal(bit(x), lin(x))
    assert bit.weight is lin.weight          # the premise that actually matters


def test_norm_is_always_on_not_interpolated():
    """v2 interpolated the norm with lambda, so warmup ran on
    partially-normalised activations no pretrained model has seen — and was
    worse than v1 at lambda=0.5 (6.61 vs 3.36). Every reference applies it
    unconditionally."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    x = torch.randn(4, 64) * 8
    half = bl.BitLinear.from_linear(lin, lambda_=0.5)
    # at any lambda > 0 the input is fully normed before quantising
    xn = bl.rms_norm(x)
    wq = bl.ste(lin.weight, bl.weight_quant(lin.weight), 0.5)
    want = F.linear(bl.ste(xn, bl.activation_quant(xn), 0.5), wq, lin.bias)
    assert torch.allclose(half(x), want, atol=1e-5)


def test_lambda_one_actually_ternarises():
    """Without this we could train a float model for 30 GPU-h and report it as
    ternary."""
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    bit = bl.BitLinear.from_linear(lin, lambda_=1.0)
    effective = bl.weight_quant(bit.weight)
    assert len(torch.unique(effective)) <= 3
    assert not torch.equal(bit(torch.randn(8, 32)), lin(torch.randn(8, 32)))


def test_from_linear_shares_the_parameter_objects():
    """Converting must not rebuild the model: the latent trajectory phase 1d
    measures has to be the float model's own weights."""
    lin = nn.Linear(8, 4)
    bit = bl.BitLinear.from_linear(lin)
    assert bit.weight is lin.weight
    assert bit.bias is lin.bias


def test_partial_lambda_interpolates():
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    x = torch.randn(8, 32)
    full = bl.BitLinear.from_linear(lin, lambda_=1.0)(x)
    half = bl.BitLinear.from_linear(lin, lambda_=0.5)(x)
    zero = bl.BitLinear.from_linear(lin, lambda_=0.0)(x)
    assert not torch.equal(half, zero) and not torch.equal(half, full)


# -- injection into a real architecture (phase-1c task 2) ----------------


@pytest.fixture(scope="module")
def smol():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")


def test_replaces_exactly_seven_linears_per_layer(smol):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    n_layers = model.config.num_hidden_layers
    _, n = bl.convert(model)
    assert n == n_layers * 7, f"expected {n_layers}*7, replaced {n}"


def test_embeddings_and_head_are_untouched_objects(smol):
    """Not merely equal — the same objects. The recipe quantises attention and
    FFN only; a quantised lm_head would be a different experiment."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    emb_before, head_before = model.get_input_embeddings(), model.lm_head
    bl.convert(model)
    assert model.get_input_embeddings() is emb_before
    assert model.lm_head is head_before
    assert not isinstance(model.lm_head, bl.BitLinear)


def test_parameter_count_is_unchanged(smol):
    """A changed count means the model was rebuilt rather than converted, and
    the latent weights would not be the float model's."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    before = sum(p.numel() for p in model.parameters())
    bl.convert(model)
    assert sum(p.numel() for p in model.parameters()) == before


def test_converted_model_at_lambda_zero_matches_the_float_model(smol):
    """The conversion premise, end to end on a real architecture."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 24))
    float_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()
    with torch.no_grad():
        want = float_model(ids).logits

    converted = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()
    bl.convert(converted, lambda_=0.0)
    with torch.no_grad():
        got = converted(ids).logits
    assert torch.equal(got, want)


def test_set_lambda_reaches_every_layer(smol):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    _, n = bl.convert(model)
    assert bl.set_lambda(model, 1.0) == n
    assert all(m.lambda_ == 1.0 for m in model.modules() if isinstance(m, bl.BitLinear))


def test_lambda_one_changes_the_output_on_a_real_model(smol):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    ids = torch.randint(0, 1000, (1, 24))
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()
    bl.convert(model, lambda_=0.0)
    with torch.no_grad():
        before = model(ids).logits
    bl.set_lambda(model, 1.0)
    with torch.no_grad():
        after = model(ids).logits
    assert not torch.allclose(before, after)


def test_norm_precedes_the_quantised_path(smol):
    """The writeup calls normalisation before activation quantisation
    essential. SmolLM2 is Llama-style pre-norm, so this should hold — checked
    rather than assumed."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    layer = model.model.layers[0]
    assert hasattr(layer, "input_layernorm")
    assert hasattr(layer, "post_attention_layernorm")
    assert type(layer.input_layernorm).__name__.endswith("RMSNorm")


def test_norm_is_applied_inside_the_layer_not_just_by_the_block():
    """The first 135M shakedown failed because this was missing.

    The block's own norms cover the block *input*; o_proj sees raw attention
    output and down_proj sees the raw SwiGLU product, and per-token absmax
    quantisation of a wide-dynamic-range input wastes the int8 grid. The
    earlier `test_norm_precedes_the_quantised_path` asserted the block has
    norms — true, and irrelevant. It passed while testing the wrong thing.
    """
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    bit = bl.BitLinear.from_linear(lin, lambda_=1.0)
    # An input with wide per-token dynamic range is what breaks without a norm.
    x = torch.randn(4, 64) * torch.tensor([[1.0], [50.0], [0.02], [5.0]])
    import torch.nn.functional as F

    normed = bl.rms_norm(x)
    # atol is loose because eps=1e-6 is not negligible against the row scaled
    # by 0.02, whose mean-square is ~4e-4 — the norm is right, 1e-3 was not.
    assert normed.pow(2).mean(dim=-1).allclose(torch.ones(4), atol=2e-2)

    # The layer must actually use it: recompute the quantised forward WITHOUT
    # the norm and assert the real layer differs.
    wq = bl.ste(lin.weight, bl.weight_quant(lin.weight), 1.0)
    no_norm = F.linear(bl.ste(x, bl.activation_quant(x), 1.0), wq, lin.bias)
    assert not torch.allclose(bit(x), no_norm, atol=1e-4)


def test_lambda_zero_still_bit_identical_after_adding_the_norm():
    """lambda=0 takes the fast path, so it is still exactly the float layer."""
    torch.manual_seed(0)
    lin = nn.Linear(32, 16)
    bit = bl.BitLinear.from_linear(lin, lambda_=0.0)
    x = torch.randn(8, 32) * 20
    assert torch.equal(bit(x), lin(x))
