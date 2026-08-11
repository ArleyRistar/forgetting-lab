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


# -- phase-2b generator fixes (2026-08-11) --------------------------------


def test_conflict_value_is_resampled_not_bumped():
    """The old collision fix took the NEXT letter, making P(v_b = v_a + 1) 25%
    instead of 1/7 — an alignment on adjacent token ids that a per-letter
    analysis reads as signal."""
    from collections import Counter
    a = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-a", "eval", n_keys=400)}
    b = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-b", "eval", n_keys=400)}
    idx = {v: i for i, v in enumerate(synthetic.VALUES)}
    adj = sum(1 for k in a if (idx[b[k]] - idx[a[k]]) % synthetic.N_VALUES == 1)
    assert 0.10 < adj / len(a) < 0.19, f"P(v2=v1+1) is {adj/len(a):.3f}, want ~1/7"


def test_conflict_values_always_differ():
    a = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-a", "eval", n_keys=200)}
    b = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-b", "eval", n_keys=200)}
    assert all(a[k] != b[k] for k in a)


def test_placebo_avoids_both_measured_values():
    """A plain permutation would give ~1/8 of keys the very letter being
    measured, and another ~1/8 no conflict at all during B."""
    a = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-a", "eval", n_keys=200)}
    b = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-conflict-b", "eval", n_keys=200)}
    p = {r["prompt"]: r["answer"]
         for r in synthetic.make("synth-placebo-a", "eval", n_keys=200)}
    assert set(p) == set(a), "placebo must cover the same keys"
    assert all(p[k] != a[k] and p[k] != b[k] for k in a)


def test_generator_seed_actually_changes_assignments():
    a0 = {r["prompt"]: r["answer"]
          for r in synthetic.make("synth-conflict-a", "eval", n_keys=200, seed=0)}
    a1 = {r["prompt"]: r["answer"]
          for r in synthetic.make("synth-conflict-a", "eval", n_keys=200, seed=1)}
    shared = set(a0) & set(a1)
    differ = sum(a0[k] != a1[k] for k in shared)
    assert differ > 0.6 * len(shared), "seeds must give different values"


def test_trace_passes_n_keys_and_seed_to_the_generator():
    """Until 2026-08-11 `trace._read` dropped both: asking for 200 keys returned
    50, and three 'seeds' shared one set of assignments in different orders."""
    from flab import trace

    ex, stats = trace.load_probe_examples("synth-conflict-a", n_eval=200,
                                          n_keys=200, gen_seed=0)
    assert stats["available"] == 200 and len(ex) == 200
    other, _ = trace.load_probe_examples("synth-conflict-a", n_eval=200,
                                         n_keys=200, gen_seed=1)
    m0 = {r["prompt"]: r["answer"] for r in ex}
    m1 = {r["prompt"]: r["answer"] for r in other}
    shared = set(m0) & set(m1)
    assert sum(m0[k] != m1[k] for k in shared) > 0.6 * len(shared)
