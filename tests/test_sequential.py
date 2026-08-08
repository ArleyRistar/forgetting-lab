"""Tests for the sequential harness (phase-1a task 4).

CPU, a tiny randomly-initialised model, 2 steps per stage. These check the
*control flow* — ordering, resume points, what lands on disk — which is where a
silently-lost boundary or a skipped stage would come from. Throughput and real
loss values are the shakedown's job (task 6).
"""
import json

import pytest
import torch

from flab import sequential, trace
from flab.runconfig import RunConfig, StageConfig, ProbeConfig
from flab.runstate import RunState, DONE, RUNNING

pytestmark = pytest.mark.skipif(
    not trace.available(), reason="TRACE archive not vendored; run scripts/fetch_trace.sh"
)


def tiny_model(tok):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    ))


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")


def cfg(**over) -> RunConfig:
    base = dict(
        run_name="t",
        stages=(StageConfig("FOMC", 2, 1e-4), StageConfig("Py150", 2, 1e-4)),
        probe=ProbeConfig(n_eval=2, max_length=128, batch_size=2),
    )
    return RunConfig(**{**base, **over})


# -- the happy path ------------------------------------------------------


def test_full_run_writes_baseline_plus_one_probe_per_boundary(tmp_path, tok):
    c = cfg()
    state = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)

    assert state.complete
    assert (tmp_path / "r" / "COMPLETE").exists()
    # N stages -> N+1 probes: the baseline is what every later one is read against.
    assert (tmp_path / "r" / "probe-baseline.json").exists()
    for i in range(len(c.stages)):
        assert (tmp_path / "r" / f"probe-after-{i}.json").exists()


def test_probe_files_cover_every_task_at_every_boundary(tmp_path, tok):
    """The N x N matrix: tasks not yet trained on are probed too, or forward
    transfer is thrown away for no saving worth having."""
    c = cfg()
    sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    for name in ("probe-baseline.json", "probe-after-0.json", "probe-after-1.json"):
        data = json.loads((tmp_path / "r" / name).read_text())
        assert set(data["tasks"]) == {"FOMC", "Py150"}
        for t in data["tasks"].values():
            assert t["n_tokens"] > 0


def test_stages_run_in_order_and_each_saves_weights(tmp_path, tok):
    c = cfg()
    state = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    assert [s.name for s in state.stages] == ["FOMC", "Py150"]
    for i, s in enumerate(state.stages):
        assert s.status == DONE
        assert s.probe == f"probe-after-{i}.json"
        assert (tmp_path / "r" / f"stage-{i}-{s.name}").is_dir()


# -- resume --------------------------------------------------------------


def test_resume_skips_completed_stages(tmp_path, tok):
    c = cfg()
    sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    before = (tmp_path / "r" / "stage-0-FOMC").stat().st_mtime

    # A second invocation on a complete run must do nothing at all.
    state = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    assert state.complete
    assert (tmp_path / "r" / "stage-0-FOMC").stat().st_mtime == before


def test_crash_between_training_and_probing_does_not_retrain(tmp_path, tok):
    """The ordering this harness is built around.

    A stage that trained and saved but died before its probe must resume into
    the *probe*, not repeat the training — and must not be silently skipped
    with its boundary missing forever.
    """
    c = cfg()
    state = RunState.create_or_load(tmp_path / "r", c)
    state.set_baseline("probe-baseline.json")
    sequential._probe_to_disk(c, tiny_model(tok), tok, tmp_path / "r", "probe-baseline.json")

    # Simulate the crash: weights saved, checkpoint recorded, still RUNNING.
    # The adapter must be genuinely written — a directory that merely exists is
    # what `checkpoint_ok` is there to reject.
    out = sequential.stage_dir(tmp_path / "r", 0, "FOMC")
    sequential._prepare_model(c, tiny_model(tok), state).save_pretrained(str(out))
    assert sequential.checkpoint_ok(out, c.mode)
    state.mark(0, RUNNING, steps_done=2, checkpoint=str(out))

    called = []
    original = sequential._train_stage
    sequential._train_stage = lambda *a, **k: called.append(1) or 2
    try:
        result = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    finally:
        sequential._train_stage = original

    assert len(called) == 1, "stage 0 must not be retrained; only stage 1 trains"
    assert result.stages[0].status == DONE
    assert (tmp_path / "r" / "probe-after-0.json").exists(), "boundary was lost"


def test_a_stage_left_running_with_no_checkpoint_is_retrained(tmp_path, tok):
    """Died mid-training with nothing saved: that stage is still work to do."""
    c = cfg()
    state = RunState.create_or_load(tmp_path / "r", c)
    state.mark(0, RUNNING, steps_done=1)          # no checkpoint recorded
    result = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    assert result.stages[0].status == DONE
    assert (tmp_path / "r" / "probe-after-0.json").exists()


def test_baseline_is_taken_once_and_not_retaken_on_resume(tmp_path, tok):
    c = cfg()
    sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    baseline = json.loads((tmp_path / "r" / "probe-baseline.json").read_text())
    mtime = (tmp_path / "r" / "probe-baseline.json").stat().st_mtime

    sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    assert (tmp_path / "r" / "probe-baseline.json").stat().st_mtime == mtime
    assert json.loads((tmp_path / "r" / "probe-baseline.json").read_text()) == baseline


def test_resuming_under_a_changed_config_is_refused(tmp_path, tok):
    from flab.runstate import ConfigMismatch

    sequential.run(cfg(), tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    with pytest.raises(ConfigMismatch):
        sequential.run(cfg(seed=7), tmp_path / "r", model=tiny_model(tok), tokenizer=tok)


# -- modes ---------------------------------------------------------------


def test_lora_mode_attaches_one_adapter_carried_across_stages(tmp_path, tok):
    c = cfg(mode="lora")
    state = RunState.create_or_load(tmp_path / "r", c)
    prepared = sequential._prepare_model(c, tiny_model(tok), state)
    assert hasattr(prepared, "peft_config"), "lora mode must attach an adapter"
    trainable = [n for n, p in prepared.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n.lower() for n in trainable)


def test_full_mode_trains_every_parameter(tmp_path, tok):
    """Phase 1c needs this path: ternary QAT has no PEFT escape hatch."""
    c = cfg(mode="full")
    state = RunState.create_or_load(tmp_path / "r", c)
    prepared = sequential._prepare_model(c, tiny_model(tok), state)
    assert not hasattr(prepared, "peft_config")
    assert all(p.requires_grad for p in prepared.parameters())


def test_both_modes_complete_a_run(tmp_path, tok):
    for mode in ("lora", "full"):
        c = cfg(mode=mode, stages=(StageConfig("FOMC", 2, 1e-4),))
        state = sequential.run(c, tmp_path / mode, model=tiny_model(tok), tokenizer=tok)
        assert state.complete, f"{mode} mode did not complete"
        assert (tmp_path / mode / "probe-after-0.json").exists()


# -- checkpoint validation -----------------------------------------------


def test_incomplete_checkpoint_is_rejected_so_the_stage_retrains(tmp_path, tok):
    """A crash *during* save_model leaves the directory there but unloadable.

    Treating directory-exists as good enough makes resume die inside peft with
    a message about a missing adapter_config, instead of simply retraining the
    stage — which is the wrong failure at 3 a.m.
    """
    half_written = tmp_path / "stage-0-FOMC"
    half_written.mkdir()
    (half_written / "README.md").write_text("partial")
    assert not sequential.checkpoint_ok(half_written, "lora")
    assert not sequential.checkpoint_ok(half_written, "full")
    assert not sequential.checkpoint_ok(tmp_path / "does-not-exist", "lora")
    assert not sequential.checkpoint_ok(None, "lora")


def test_run_retrains_past_an_unloadable_checkpoint(tmp_path, tok):
    c = cfg(stages=(StageConfig("FOMC", 2, 1e-4),))
    state = RunState.create_or_load(tmp_path / "r", c)
    junk = sequential.stage_dir(tmp_path / "r", 0, "FOMC")
    junk.mkdir(parents=True)
    (junk / "README.md").write_text("partial")
    state.mark(0, RUNNING, steps_done=1, checkpoint=str(junk))

    result = sequential.run(c, tmp_path / "r", model=tiny_model(tok), tokenizer=tok)
    assert result.complete
    assert sequential.checkpoint_ok(result.stages[0].checkpoint, c.mode)
