"""Tests for run config and crash-resume state (phase-1a task 2).

CPU-only and fast on purpose — these must stay cheap enough that `uv run
pytest` remains a pre-commit habit rather than something done occasionally.
"""
import json

import pytest

from flab.runconfig import RunConfig, StageConfig, ProbeConfig
from flab.runstate import RunState, StageState, ConfigMismatch, DONE, RUNNING, FAILED

DEV = "configs/dev-3stage.yaml"


def cfg(**over) -> RunConfig:
    base = dict(
        run_name="t",
        stages=(StageConfig("FOMC", max_steps=10, learning_rate=1e-4), StageConfig("Py150", max_steps=10, learning_rate=1e-4)),
        probe=ProbeConfig(),
    )
    return RunConfig(**{**base, **over})


# -- config --------------------------------------------------------------


def test_dev_config_loads_and_validates():
    c = RunConfig.load(DEV)
    assert [s.task for s in c.stages] == ["FOMC", "Py150", "ScienceQA"]
    assert c.mode == "lora"
    assert c.seed == 0
    assert c.probe.max_length == 1024


def test_probe_all_resolves_to_stage_tasks_not_all_nine():
    c = RunConfig.load(DEV)
    assert c.probe_tasks == ["FOMC", "Py150", "ScienceQA"]


def test_probe_accepts_explicit_list_for_forward_transfer():
    c = cfg(probe=ProbeConfig(tasks=["FOMC", "ScienceQA"]))
    assert c.probe_tasks == ["FOMC", "ScienceQA"]


def test_probe_all_dedupes_repeated_tasks():
    c = cfg(stages=(StageConfig("FOMC", max_steps=10, learning_rate=1e-4), StageConfig("FOMC", max_steps=10, learning_rate=1e-4)))
    assert c.probe_tasks == ["FOMC"]


@pytest.mark.parametrize("bad", [
    dict(mode="ternary"),                                   # not yet a mode; 1c adds it
    dict(stages=()),                                        # a run with no stages
    dict(stages=(StageConfig("NotATask", max_steps=10, learning_rate=1e-4),)),
    dict(stages=(StageConfig("FOMC", max_steps=0, learning_rate=1e-4),)),           # zero steps
    dict(stages=(StageConfig("FOMC", max_steps=10, learning_rate=0.0),)),           # zero LR
    dict(probe=ProbeConfig(n_eval=0)),
    dict(probe=ProbeConfig(batch_size=0)),
    dict(probe=ProbeConfig(tasks=["NotATask"])),
])
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        cfg(**bad).validate()


def test_hash_is_stable_and_order_independent():
    assert cfg().content_hash == cfg().content_hash
    a = cfg(probe=ProbeConfig(tasks=["FOMC"], n_eval=8))
    b = cfg(probe=ProbeConfig(n_eval=8, tasks=["FOMC"]))
    assert a.content_hash == b.content_hash


def test_hash_changes_when_the_experiment_changes():
    base = cfg().content_hash
    assert cfg(seed=1).content_hash != base
    assert cfg(mode="full").content_hash != base
    assert cfg(optim="adamw_bnb_8bit").content_hash != base
    assert cfg(stages=(StageConfig("FOMC", max_steps=11, learning_rate=1e-4),)).content_hash != base
    # batch_size changes the numbers, so it must change the hash
    assert cfg(probe=ProbeConfig(batch_size=2)).content_hash != base


# -- state ---------------------------------------------------------------


def test_fresh_run_starts_at_stage_zero_and_writes_provenance(tmp_path):
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    assert st.next_index() == 0
    prov = json.loads((tmp_path / "r" / "run.json").read_text())
    assert prov["config_hash"] == c.content_hash
    assert "git_commit" in prov and "git_dirty" in prov


def test_resume_picks_up_at_the_first_unfinished_stage(tmp_path):
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    st.mark(0, DONE, steps_done=10)
    # Simulate the process dying: a brand-new object off the same directory.
    again = RunState.create_or_load(tmp_path / "r", c)
    assert again.next_index() == 1
    assert again.stages[0].status == DONE


def test_a_stage_left_running_is_resumed_not_skipped(tmp_path):
    """The 3 a.m. case: killed mid-stage, so that stage is still work to do."""
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    st.mark(0, RUNNING, steps_done=4)
    assert RunState.create_or_load(tmp_path / "r", c).next_index() == 0


def test_a_failed_stage_is_retried_not_treated_as_terminal(tmp_path):
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    st.mark(0, FAILED)
    # If FAILED were terminal the supervisor's retry loop could never advance.
    assert RunState.create_or_load(tmp_path / "r", c).next_index() == 0


def test_completion(tmp_path):
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    for i in range(len(c.stages)):
        st.mark(i, DONE)
    assert st.complete and st.next_index() is None


def test_resuming_under_a_changed_config_is_refused(tmp_path):
    """The failure this file exists to prevent."""
    RunState.create_or_load(tmp_path / "r", cfg())
    with pytest.raises(ConfigMismatch):
        RunState.create_or_load(tmp_path / "r", cfg(seed=99))


def test_state_survives_a_torn_write(tmp_path):
    """Atomicity: a stray temp file must not be mistaken for the state file."""
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    st.mark(0, DONE)
    (tmp_path / "r" / "runstate.json.tmp").write_text("{ truncated garbage")
    assert RunState.create_or_load(tmp_path / "r", c).next_index() == 1


def test_mark_rejects_unknown_fields(tmp_path):
    st = RunState.create_or_load(tmp_path / "r", cfg())
    with pytest.raises(AttributeError):
        st.mark(0, DONE, stpes_done=10)  # typo must not vanish into the void


def test_baseline_probe_round_trips(tmp_path):
    c = cfg()
    st = RunState.create_or_load(tmp_path / "r", c)
    st.set_baseline("probe-baseline.json")
    assert RunState.create_or_load(tmp_path / "r", c).baseline_probe == "probe-baseline.json"


# -- phase-1b: parameterisation ------------------------------------------


def test_paper_calibration_config_loads_with_their_protocol():
    """configs/calib-paper.yaml is the executable record of what we replicated,
    so drift in it is a silent invalidation of the calibration."""
    c = RunConfig.load("configs/calib-paper.yaml")
    assert c.trace_variant == "LLM-CL-Benchmark_500"
    assert [s.task for s in c.stages] == ["FOMC", "ScienceQA", "NumGLUE-cm"]
    assert all(s.epochs == 1 and s.max_steps is None for s in c.stages)
    assert all(s.learning_rate == 5e-5 for s in c.stages)
    assert (c.lora.r, c.lora.alpha, c.lora.dropout) == (8, 16, 0.05)
    assert c.lora.target_modules == "all-linear"
    assert (c.train.batch_size, c.train.grad_accum) == (2, 8)   # 16 effective
    assert c.train.max_length == 512 and c.probe.max_length == 512
    assert c.model_sha256 == "1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f"


def test_exactly_one_schedule_per_stage():
    with pytest.raises(ValueError, match="exactly one"):
        cfg(stages=(StageConfig("FOMC", learning_rate=1e-4),)).validate()
    with pytest.raises(ValueError, match="exactly one"):
        cfg(stages=(StageConfig("FOMC", learning_rate=1e-4, max_steps=5, epochs=1),)).validate()
    cfg(stages=(StageConfig("FOMC", learning_rate=1e-4, epochs=1),)).validate()


def test_unknown_trace_variant_rejected():
    with pytest.raises(ValueError, match="variant"):
        cfg(trace_variant="LLM-CL-Benchmark_9999").validate()


def test_target_modules_accepts_all_linear_or_a_list_only():
    from flab.runconfig import LoraSpec

    cfg(lora=LoraSpec(target_modules="all-linear")).validate()
    cfg(lora=LoraSpec(target_modules=("q_proj",))).validate()
    with pytest.raises(ValueError, match="all-linear"):
        cfg(lora=LoraSpec(target_modules="q_proj")).validate()


def test_lora_and_train_shape_are_in_the_hash():
    """Changing what is adapted, or how much, changes the numbers."""
    from flab.runconfig import LoraSpec, TrainSpec

    base = cfg().content_hash
    assert cfg(lora=LoraSpec(r=8)).content_hash != base
    assert cfg(lora=LoraSpec(target_modules="all-linear")).content_hash != base
    assert cfg(train=TrainSpec(max_length=512)).content_hash != base
    assert cfg(train=TrainSpec(grad_accum=8)).content_hash != base
    assert cfg(trace_variant="LLM-CL-Benchmark_500").content_hash != base


def test_digest_mismatch_refuses_to_run(tmp_path, monkeypatch):
    """A mirror that stops being byte-identical must fail, not warn."""
    from flab import sequential

    monkeypatch.setattr(sequential, "verify_model_digest", sequential.verify_model_digest)
    c = cfg(model_sha256="0" * 64)
    import huggingface_hub
    fake = tmp_path / "model.safetensors"
    fake.write_bytes(b"not the real weights")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(fake))
    with pytest.raises(RuntimeError, match="do not match the pinned digest"):
        sequential.verify_model_digest(c)


def test_no_digest_pinned_skips_the_check():
    from flab import sequential

    assert sequential.verify_model_digest(cfg()) is None
