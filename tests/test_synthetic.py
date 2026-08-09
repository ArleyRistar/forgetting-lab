"""Tests for the synthetic forgetting controls (phase-1b task 5).

These guard the *premises* of the analytic argument. If conflict-a and
conflict-b ever agreed on a key, or disjoint's keys overlapped, the control
would still produce numbers — they just would not mean what the gate claims.
"""
import math

import pytest

from flab import synthetic, trace


def test_conflict_shares_every_key_and_disagrees_on_every_value():
    """The premise of 'forgetting is logically forced'."""
    a = {r["prompt"]: r["answer"] for r in synthetic.make("synth-conflict-a", "eval")}
    b = {r["prompt"]: r["answer"] for r in synthetic.make("synth-conflict-b", "eval")}
    assert set(a) == set(b), "conflict sides must ask about the same keys"
    assert all(a[k] != b[k] for k in a), "a shared key that agreed would dilute the control"
    assert synthetic.conflicting_values() == 50


def test_disjoint_shares_no_key_at_all():
    """The premise of 'zero forgetting is achievable'."""
    a = {r["prompt"] for r in synthetic.make("synth-disjoint-a", "eval")}
    b = {r["prompt"] for r in synthetic.make("synth-disjoint-b", "eval")}
    assert a and b and not (a & b)


def test_values_come_from_the_declared_set_so_chance_is_known():
    for task in synthetic.TASKS:
        vals = {r["answer"] for r in synthetic.make(task, "eval")}
        assert vals <= set(synthetic.VALUES)
    assert synthetic.chance_nll() == pytest.approx(math.log(8))


def test_train_repeats_keys_and_eval_does_not():
    tr = synthetic.make("synth-conflict-a", "train", n_keys=10, repeats=4)
    ev = synthetic.make("synth-conflict-a", "eval", n_keys=10)
    assert len(tr) == 40 and len(ev) == 10
    assert len({r["prompt"] for r in tr}) == 10        # same keys, repeated
    assert len({r["prompt"] for r in ev}) == 10


def test_generation_is_deterministic():
    assert synthetic.make("synth-conflict-a", "eval") == synthetic.make("synth-conflict-a", "eval")


def test_unknown_synthetic_task_rejected():
    with pytest.raises(ValueError):
        synthetic.make("synth-nope-a", "eval")


def test_answers_are_single_tokens_so_nll_is_one_clean_association():
    """One answer token per example keeps the analytic scale exact: the NLL is
    the log-probability of the association itself, not of a spelling."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    for v in synthetic.VALUES:
        assert len(tok(v, add_special_tokens=False)["input_ids"]) == 1, v


def test_synthetic_tasks_load_through_the_normal_trace_path():
    """They must run through the same loader as real tasks, or the control
    validates a different instrument than phase 2 uses."""
    assert set(synthetic.TASKS) <= set(trace.TASKS)
    ex, stats = trace.load_probe_examples("synth-disjoint-a", n_eval=5, seed=0)
    assert len(ex) == 5 and stats["available"] == 50
    dd = trace.load_task("synth-conflict-b", n_train=8, n_eval=4)
    assert len(dd["train"]) == 8 and dd["train"][0]["text"]
