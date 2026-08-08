"""Tests for the NLL probe (phase-1a task 3).

CPU-only with a tiny random model — these check the probe's *arithmetic and
boundaries*, which is where a silent corruption of the loss matrix would come
from. The real-model cost measurement is a separate GPU step.
"""
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
