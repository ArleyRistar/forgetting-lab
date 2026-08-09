"""Tests for the NLL probe (phase-1a task 3).

CPU-only with a tiny random model — these check the probe's *arithmetic and
boundaries*, which is where a silent corruption of the loss matrix would come
from. The real-model cost measurement is a separate GPU step.
"""
import math

import pytest
import torch

from flab import probes, trace

pytestmark = pytest.mark.skipif(
    not trace.available(), reason="TRACE archive not vendored; run scripts/fetch_trace.sh"
)


@pytest.fixture(scope="module")
def tiny():
    """A small randomly-initialised causal LM plus the real tokenizer."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval(), tok


# -- encoding boundaries -------------------------------------------------


def test_labels_mask_the_prompt_and_keep_the_whole_answer(tiny):
    _, tok = tiny
    ids, labels, truncated = probes._encode(tok, "What is 2+2?", "four", 1024)
    answer_ids = tok("four", add_special_tokens=False)["input_ids"]
    assert not truncated
    assert labels.count(probes.IGNORE) == len(ids) - len(answer_ids)
    assert labels[-len(answer_ids):] == answer_ids   # answer intact, at the end
    assert ids[-len(answer_ids):] == answer_ids


def test_long_prompt_is_left_truncated_answer_untouched(tiny):
    _, tok = tiny
    answer = "the answer"
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids, labels, truncated = probes._encode(tok, "word " * 5000, answer, 128)
    assert truncated
    assert len(ids) <= 128
    assert labels[-len(answer_ids):] == answer_ids   # never sacrificed


def test_example_whose_answer_alone_overflows_is_dropped_not_mangled(tiny):
    _, tok = tiny
    ids, labels, _ = probes._encode(tok, "hi", "word " * 5000, 128)
    assert ids is None and labels is None


# -- measurement ---------------------------------------------------------


def test_probe_reports_per_task_numbers_and_no_cross_task_mean(tiny):
    model, tok = tiny
    out = probes.probe_all(model, tok, ["FOMC", "ScienceQA"], n_eval=4, max_length=256)
    assert set(out["tasks"]) == {"FOMC", "ScienceQA"}
    # A single aggregate NLL would be meaningless across answer-length regimes.
    assert "nll" not in out
    for t in ("FOMC", "ScienceQA"):
        r = out["tasks"][t]
        assert r["n_tokens"] > 0
        assert r["nll"] > 0
        assert 0.0 <= r["token_acc"] <= 1.0


def test_answer_length_regimes_differ_as_expected(tiny):
    """FOMC answers are one letter; ScienceQA answers run ~200 tokens."""
    model, tok = tiny
    out = probes.probe_all(model, tok, ["FOMC", "ScienceQA"], n_eval=4, max_length=512)
    fomc = out["tasks"]["FOMC"]["n_tokens"]
    sci = out["tasks"]["ScienceQA"]["n_tokens"]
    assert fomc <= 8, f"FOMC should score ~1 token/example, got {fomc} over 4"
    assert sci > fomc * 5, "ScienceQA should dominate FOMC on token count"


def test_nll_matches_a_hand_rolled_forward_pass(tiny):
    """Guard the arithmetic itself: shift, mask and normalisation."""
    import torch.nn.functional as F

    model, tok = tiny
    ex, _ = trace.load_probe_examples("FOMC", n_eval=1, seed=0)
    ids, labels, _ = probes._encode(tok, ex[0]["prompt"], ex[0]["answer"], 1024)

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = torch.tensor([labels])[:, 1:]
    mask = shift_labels != probes.IGNORE
    expected = F.cross_entropy(shift_logits[mask], shift_labels[mask]).item()

    got = probes.probe_task(model, tok, "FOMC", n_eval=1, max_length=1024, batch_size=1)
    assert got.nll == pytest.approx(expected, rel=1e-4)


def test_batching_does_not_change_the_result(tiny):
    """Padding must not leak into the aggregate."""
    model, tok = tiny
    a = probes.probe_task(model, tok, "FOMC", n_eval=8, max_length=512, batch_size=1)
    b = probes.probe_task(model, tok, "FOMC", n_eval=8, max_length=512, batch_size=8)
    assert a.n_tokens == b.n_tokens
    assert a.nll == pytest.approx(b.nll, rel=1e-3)


def test_same_seed_probes_the_same_examples(tiny):
    """Every boundary must score the identical held-out set, or the matrix
    compares different data rather than different models."""
    model, tok = tiny
    a = probes.probe_task(model, tok, "Py150", n_eval=8, max_length=512, seed=0)
    b = probes.probe_task(model, tok, "Py150", n_eval=8, max_length=512, seed=0)
    c = probes.probe_task(model, tok, "Py150", n_eval=8, max_length=512, seed=1)
    assert a.n_tokens == b.n_tokens and a.nll == pytest.approx(b.nll)
    assert c.n_tokens != a.n_tokens or c.nll != pytest.approx(a.nll)


def test_clamped_n_eval_raises_a_warning_not_a_silent_short_run(tiny):
    model, tok = tiny
    r = probes.probe_task(model, tok, "NumGLUE-cm", n_eval=10_000, max_length=256)
    assert r.n_examples < 10_000
    assert r.warning and "exist" in r.warning


def test_probe_restores_training_mode(tiny):
    model, tok = tiny
    model.train()
    try:
        probes.probe_task(model, tok, "FOMC", n_eval=2, max_length=256)
        assert model.training, "probe must not leave the model in eval mode"
    finally:
        model.eval()


# -- the generative-eval guard -------------------------------------------


def test_live_adapter_is_refused_for_generation(tiny):
    model, _ = tiny

    class FakePeft:
        peft_config = {}

    with pytest.raises(RuntimeError, match="1.89"):
        probes.ensure_no_live_adapter(FakePeft())
    probes.ensure_no_live_adapter(model)  # a plain model is fine


# -- stability probe (phase-1b task 3) -----------------------------------


def lora_wrap(model):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "v_proj"],
    ))


def test_fresh_adapter_gives_exactly_zero_kl(tiny):
    """The decisive validation of the drift metric.

    peft initialises lora_B to zeros, so an untrained adapter is a mathematical
    no-op: the adapter-enabled and adapter-disabled forward passes are the same
    function. KL must therefore be 0. If it is not, the two passes are not
    comparable and every drift number afterwards is noise rather than signal.
    """
    model, tok = tiny
    r = probes.probe_stability(lora_wrap(model), tok, n_ref=4, max_length=256, batch_size=2)
    assert r.warning is None and r.n_tokens > 0
    assert r.kl_from_base == pytest.approx(0.0, abs=1e-6), r.kl_from_base
    assert r.delta_entropy == pytest.approx(0.0, abs=1e-6)
    assert r.margin == pytest.approx(r.base_margin, abs=1e-6)


def test_a_perturbed_adapter_gives_nonzero_kl(tiny):
    """...and that it can actually see drift, so the zero above is not vacuous."""
    import torch

    model, tok = tiny
    peft_model = lora_wrap(model)
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.05)
    r = probes.probe_stability(peft_model, tok, n_ref=4, max_length=256, batch_size=2)
    assert r.kl_from_base > 1e-4, f"perturbed adapter should drift, got {r.kl_from_base}"


def test_lima_eval_is_all_empty_answers_and_yields_a_warning_not_a_number(tiny):
    """Regression test for a real data trap.

    Lima is the obvious reference set — TRACE's replay split, disjoint from
    every task — but its eval/test splits are 100% empty answers. Reaching for
    Lima/eval must produce a refusal, never a plausible-looking drift number.
    """
    model, tok = tiny
    r = probes.probe_stability(lora_wrap(model), tok, n_ref=8, max_length=256, split="eval")
    assert r.n_tokens == 0
    assert r.kl_from_base is None
    assert r.warning and "nothing was measured" in r.warning


def test_stability_refuses_a_bare_model_with_no_base(tiny):
    """Full fine-tuning has no adapter to disable, so phase 1c must pass a base
    explicitly. Silently comparing a model to itself would report zero drift."""
    model, tok = tiny
    r = probes.probe_stability(model, tok, n_ref=4, max_length=256)
    assert r.kl_from_base is None
    assert r.warning and "not a PEFT model" in r.warning


def test_reference_set_selection_is_deterministic():
    a, _ = trace.load_reference_examples(n=8, seed=0)
    b, _ = trace.load_reference_examples(n=8, seed=0)
    c, _ = trace.load_reference_examples(n=8, seed=1)
    assert a == b and a != c


def test_both_kl_directions_are_zero_for_a_fresh_adapter(tiny):
    model, tok = tiny
    r = probes.probe_stability(lora_wrap(model), tok, n_ref=4, max_length=256, batch_size=2)
    assert r.kl_from_base == pytest.approx(0.0, abs=1e-6)
    assert r.kl_to_base == pytest.approx(0.0, abs=1e-6)


def test_kl_arithmetic_against_hand_computed_distributions():
    """Test the formula directly, not through a model.

    2606.27634 defines drift as KL(p_k || p_0) and KL is asymmetric, so the
    direction matters. It cannot be verified through the tiny test model: its
    output is near-uniform, KL saturates around 0.026 even under an 8x weight
    perturbation, and both directions coincide because KL is symmetric to
    leading order for small divergences. A direction bug would pass silently.
    """
    import torch

    lp_cur = torch.tensor([[0.9, 0.1]]).log()
    lp_base = torch.tensor([[0.5, 0.5]]).log()
    kl_f, kl_r, dh = probes.kl_pair(lp_cur, lp_base)

    # KL(cur||base) = .9 ln(.9/.5) + .1 ln(.1/.5)
    assert kl_f.item() == pytest.approx(0.9 * math.log(1.8) + 0.1 * math.log(0.2), rel=1e-5)
    # KL(base||cur) = .5 ln(.5/.9) + .5 ln(.5/.1)
    assert kl_r.item() == pytest.approx(0.5 * math.log(5 / 9) + 0.5 * math.log(5), rel=1e-5)
    assert kl_f.item() != pytest.approx(kl_r.item(), rel=1e-2), "the two directions must differ"
    # H(cur) - H(base): a sharper current distribution has lower entropy
    h_cur = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    h_base = math.log(2)
    assert dh.item() == pytest.approx(h_cur - h_base, rel=1e-5)
    assert dh.item() < 0


def test_kl_pair_is_zero_for_identical_distributions():
    import torch

    lp = torch.tensor([[0.3, 0.7]]).log()
    kl_f, kl_r, dh = probes.kl_pair(lp, lp)
    assert kl_f.item() == pytest.approx(0.0, abs=1e-7)
    assert kl_r.item() == pytest.approx(0.0, abs=1e-7)
    assert dh.item() == pytest.approx(0.0, abs=1e-7)
