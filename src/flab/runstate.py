"""Crash-resume state for a sequential run (spec §6 1a, phase-1a task 2).

Spec §6 1a wants crash-resume "from day one": at 5-10 h/week of human
attention, a 3 a.m. OOM must cost minutes, not a calendar day. That needs two
levels, and this file is the outer one — which stage are we on. The inner level
(where inside a stage) is HF Trainer's `resume_from_checkpoint`.

Two invariants carry the safety:

  * **Atomic writes.** A crash *during* the write that records progress must
    not destroy the file that makes recovery possible. Write to a temp file in
    the same directory, then `os.replace`, which is atomic on POSIX.
  * **Config-hash guard.** Resuming an existing run directory under a changed
    config would splice two experiments together silently. That is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import os

from flab.runconfig import RunConfig, provenance

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


class ConfigMismatch(RuntimeError):
    """Raised when a run directory was built from a different config."""


@dataclass
class StageState:
    name: str
    status: str = PENDING
    steps_done: int = 0
    checkpoint: str | None = None
    probe: str | None = None


@dataclass
class RunState:
    run_dir: Path
    config_hash: str
    stages: list[StageState] = field(default_factory=list)
    baseline_probe: str | None = None

    # -- persistence -----------------------------------------------------

    @property
    def path(self) -> Path:
        return self.run_dir / "runstate.json"

    def save(self) -> None:
        payload = {
            "config_hash": self.config_hash,
            "baseline_probe": self.baseline_probe,
            "stages": [asdict(s) for s in self.stages],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)  # atomic; a torn write cannot survive

    @classmethod
    def create_or_load(cls, run_dir: str | Path, cfg: RunConfig) -> RunState:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "runstate.json"

        if state_path.is_file():
            data = json.loads(state_path.read_text())
            if data.get("config_hash") != cfg.content_hash:
                raise ConfigMismatch(
                    f"{run_dir} was built from config {data.get('config_hash', '?')[:12]} "
                    f"but this config hashes to {cfg.content_hash[:12]}. Resuming would "
                    f"splice two different experiments into one set of numbers. "
                    f"Use a new run_name, or delete the directory deliberately."
                )
            state = cls(
                run_dir=run_dir,
                config_hash=data["config_hash"],
                stages=[StageState(**s) for s in data["stages"]],
                baseline_probe=data.get("baseline_probe"),
            )
            # A stage left RUNNING means the process died inside it. That is
            # resumable, not failed - the inner checkpoint decides where.
            return state

        state = cls(
            run_dir=run_dir,
            config_hash=cfg.content_hash,
            stages=[StageState(name=s.task) for s in cfg.stages],
        )
        (run_dir / "run.json").write_text(json.dumps(provenance(cfg), indent=2))
        state.save()
        return state

    # -- progress --------------------------------------------------------

    def next_index(self) -> int | None:
        """First stage not yet finished, or None when the run is complete.

        Anything that is not DONE is work still to do — including RUNNING (died
        mid-stage) and FAILED (retry). Treating FAILED as terminal here would
        make the supervisor's retry loop unable to make progress.
        """
        for i, s in enumerate(self.stages):
            if s.status != DONE:
                return i
        return None

    def mark(self, index: int, status: str, **fields) -> None:
        stage = self.stages[index]
        stage.status = status
        for k, v in fields.items():
            if not hasattr(stage, k):
                raise AttributeError(f"StageState has no field {k!r}")
            setattr(stage, k, v)
        self.save()

    def set_baseline(self, probe_file: str) -> None:
        self.baseline_probe = probe_file
        self.save()

    @property
    def complete(self) -> bool:
        return self.next_index() is None
