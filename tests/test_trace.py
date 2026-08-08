"""Tests for the TRACE loader.

These skip rather than fail when `data/trace/` is absent, so a fresh clone
still passes `uv run pytest` before anyone has fetched the 77 MB archive.
"""
import pytest

from flab import trace

pytestmark = pytest.mark.skipif(
    not trace.available(), reason="TRACE archive not vendored; run scripts/fetch_trace.sh"
)


def test_every_task_loads():
    for name in trace.TASKS:
        dd = trace.load_task(name, n_train=8, n_eval=4)
        assert len(dd["train"]) == 8
        assert len(dd["eval"]) <= 4
        assert dd["train"][0]["text"]


def test_dev_tasks_are_known_tasks():
    assert set(trace.DEV_TASKS) <= set(trace.TASKS)


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        trace.load_task("NotATask")


def test_n_eval_is_clamped_not_silently_short():
    # NumGLUE-cm ships 41 eval examples; asking for 200 must clamp AND say so,
    # because a probe quietly running on a fifth of what was requested is the
    # kind of number that ends up in a paper.
    dd = trace.load_task("NumGLUE-cm", n_train=4, n_eval=200)
    assert dd.stats["eval"]["used"] == dd.stats["eval"]["available"] < 200
    assert dd.stats["eval"]["requested"] == 200


def test_format_puts_answer_last_and_tagged():
    text = trace.format_example("Q?", "A!")
    assert text.index("Q?") < text.index("A!")
    assert "<|user|>" in text and "<|assistant|>" in text
    assert text.rstrip().endswith("<|end|>")


def test_splits_are_disjoint():
    dd = trace.load_task("FOMC", n_train=64, n_eval=64)
    assert not (set(dd["train"]["text"]) & set(dd["eval"]["text"]))


def test_seed_is_deterministic():
    a = trace.load_task("FOMC", n_train=16, n_eval=4, seed=0)["train"]["text"]
    b = trace.load_task("FOMC", n_train=16, n_eval=4, seed=0)["train"]["text"]
    c = trace.load_task("FOMC", n_train=16, n_eval=4, seed=1)["train"]["text"]
    assert a == b
    assert a != c


def test_truncation_never_eats_the_answer():
    """The property the whole measurement rests on."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
    # Py150 has the long tail among the dev tasks: p95 ~9k chars at seq 1024.
    dd = trace.load_task("Py150", n_train=200, n_eval=8, tokenizer=tok, max_length=1024)
    assert dd.stats["train"]["prompts_truncated"] > 0, "expected Py150 to exercise truncation"
    for text in dd["train"]["text"]:
        assert text.rstrip().endswith("<|end|>"), "answer was cut by the window"
        assert len(tok(text)["input_ids"]) <= 1024 + 8  # small slack for tag tokens
