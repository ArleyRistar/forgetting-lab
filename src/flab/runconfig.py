"""Declarative run configuration and provenance (spec §6 1a, phase-1a task 2).

A sequential run is a YAML file, not a pile of CLI flags, for one reason: the
phase-1 deliverable is a rig "re-runnable from a commit hash", and that is only
true if the thing the commit hash points at fully determines the experiment.

`content_hash` is what makes resume safe. Silently continuing an existing run
directory under a *different* config would splice two experiments into one set
of numbers, and nothing downstream would ever reveal it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import json
import subprocess

import yaml

from flab import trace

MODES = ("lora", "full")
REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StageConfig:
    task: str
    max_steps: int
    learning_rate: float


@dataclass(frozen=True)
class ProbeConfig:
    # "all" means every task named in `stages` — not all nine vendored sets.
    # An explicit list also probes tasks the run never trains on, which is pure
    # forward transfer and costs only forward passes.
    tasks: str | list[str] = "all"
    n_eval: int = 200
    max_length: int = 1024


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    model: str = "HuggingFaceTB/SmolLM2-360M"
    seed: int = 0
    mode: str = "lora"
    optim: str = "adamw_torch"
    stages: tuple[StageConfig, ...] = ()
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        stages = tuple(StageConfig(**s) for s in raw.pop("stages", []))
        probe = ProbeConfig(**(raw.pop("probe", {}) or {}))
        cfg = cls(stages=stages, probe=probe, **raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not self.stages:
            raise ValueError("a run needs at least one stage")
        for s in self.stages:
            if s.task not in trace.TASKS:
                raise ValueError(f"unknown task {s.task!r}; known: {trace.TASKS}")
            if s.max_steps <= 0:
                raise ValueError(f"{s.task}: max_steps must be positive")
            if s.learning_rate <= 0:
                raise ValueError(f"{s.task}: learning_rate must be positive")
        if self.probe.n_eval <= 0:
            raise ValueError("probe.n_eval must be positive")
        for t in self.probe_tasks:
            if t not in trace.TASKS:
                raise ValueError(f"unknown probe task {t!r}")

    # -- derived ---------------------------------------------------------

    @property
    def probe_tasks(self) -> list[str]:
        """Resolve `probe.tasks`, preserving stage order and dropping repeats."""
        if self.probe.tasks == "all":
            seen: dict[str, None] = {}
            for s in self.stages:
                seen.setdefault(s.task, None)
            return list(seen)
        if isinstance(self.probe.tasks, str):
            raise ValueError("probe.tasks must be 'all' or a list of task names")
        return list(self.probe.tasks)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def content_hash(self) -> str:
        """Stable sha256 over the whole config.

        Key order is canonicalised so that reordering the YAML does not look
        like a different experiment. Nothing time- or path-dependent is mixed
        in, or the hash would differ on every run and the resume guard would be
        useless.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# -- provenance ----------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def trace_checksum() -> str | None:
    """sha256 of the vendored TRACE archive, if it was fetched by the script."""
    p = REPO / "data" / "TRACE-Benchmark.zip.sha256"
    return p.read_text().split()[0] if p.is_file() else None


def provenance(cfg: RunConfig) -> dict:
    """Everything needed to say what produced a number.

    `git_dirty` is recorded rather than enforced: refusing to run on a dirty
    tree would be the wrong trade during development, but a result whose
    provenance is a commit hash *plus uncommitted edits* must say so.
    """
    return {
        "config_hash": cfg.content_hash,
        "config": cfg.to_dict(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "trace_sha256": trace_checksum(),
    }
