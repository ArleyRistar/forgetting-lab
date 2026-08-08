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
        stages=(StageConfig("FOMC", 10, 1e-4), StageConfig("Py150", 10, 1e-4)),
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
    c = cfg(stages=(StageConfig("FOMC", 10, 1e-4), StageConfig("FOMC", 10, 1e-4)))
    assert c.probe_tasks == ["FOMC"]


@pytest.mark.parametrize("bad", [
    dict(mode="ternary"),                                   # not yet a mode; 1c adds it
    dict(stages=()),                                        # a run with no stages
    dict(stages=(StageConfig("NotATask", 10, 1e-4),)),
    dict(stages=(StageConfig("FOMC", 0, 1e-4),)),           # zero steps
    dict(stages=(StageConfig("FOMC", 10, 0.0),)),           # zero LR
    dict(probe=ProbeConfig(n_eval=0)),
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
    assert cfg(stages=(StageConfig("FOMC", 11, 1e-4),)).content_hash != base


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
