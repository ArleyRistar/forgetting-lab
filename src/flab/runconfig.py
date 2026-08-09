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


DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class StageConfig:
    task: str
    learning_rate: float
    # Exactly one of these. 1a fixed a step count; 2606.27634 trains one epoch
    # per task, and at 500 examples those are very different amounts of training.
    max_steps: int | None = None
    epochs: float | None = None


@dataclass(frozen=True)
class LoraSpec:
    """LoRA shape. 1a hardcoded r=16/α=32 over seven named modules; the paper
    uses r=8/α=16 over `all-linear`, which is a *different adapted set*, not
    just a smaller one."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: str | tuple[str, ...] = DEFAULT_LORA_TARGETS


@dataclass(frozen=True)
class TrainSpec:
    """Training shape, separate from probe shape so each is explicit in the
    hash. `max_length` here and in `probe` normally match — a probe seeing more
    context than training did is legitimate but is a different measurement, so
    it should be a choice rather than an accident."""
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 1024
    lr_scheduler: str = "cosine"
    warmup_steps: int | None = None   # None -> min(20, max(1, steps // 10))


@dataclass(frozen=True)
class ProbeConfig:
    # "all" means every task named in `stages` — not all nine vendored sets.
    # An explicit list also probes tasks the run never trains on, which is pure
    # forward transfer and costs only forward passes.
    tasks: str | list[str] = "all"
    n_eval: int = 200
    max_length: int = 1024
    # Part of the experiment, not a performance knob. The probe is bit-exact
    # at a fixed batch size but NLL shifts with it (measured 2026-08-08: FOMC
    # moves 0.0065 between batch 2 and 4, from bf16 reduction order changing
    # with padding width). Changing it mid-run would manufacture a shift that
    # reads as forgetting, so it lives in the hash.
    batch_size: int = 4
    # Reference-set size for the stability probe (2606.27634's R). 0 disables it.
    reference_n: int = 200


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    model: str = "HuggingFaceTB/SmolLM2-360M"
    # Pin the weights, not just the name. Llama 3.2 1B is gated upstream and we
    # use a byte-identical mirror; a mirror silently ceasing to be identical is
    # the failure that matters, so the digest is checked on load.
    model_sha256: str | None = None
    trace_variant: str = "LLM-CL-Benchmark_5000"
    # Replication knobs. All hashed: each changes every number a run produces.
    #   prompt_style  "flab" (ours) | "paper" (2606.27634's chat-template render)
    #   kl_scope      "answer_tokens" (ours) | "next_token" (theirs: ONE position
    #                 per example, right after the prompt — the single biggest
    #                 source of our 4-8x KL gap)
    #   reference     "lima" (ours) | "task_eval" (theirs: 20% carved from each
    #                 task's eval split, 48 examples for the _500 variant)
    #   eval_split    "eval" (ours) | "test" (theirs)
    prompt_style: str = "flab"
    kl_scope: str = "answer_tokens"
    reference: str = "lima"
    eval_split: str = "eval"
    seed: int = 0
    mode: str = "lora"
    optim: str = "adamw_torch"
    stages: tuple[StageConfig, ...] = ()
    lora: LoraSpec = field(default_factory=LoraSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    probe: ProbeConfig = field(default_factory=ProbeConfig)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        stages = tuple(StageConfig(**s) for s in raw.pop("stages", []))
        lora_raw = raw.pop("lora", {}) or {}
        if isinstance(lora_raw.get("target_modules"), list):
            lora_raw["target_modules"] = tuple(lora_raw["target_modules"])
        cfg = cls(
            stages=stages,
            lora=LoraSpec(**lora_raw),
            train=TrainSpec(**(raw.pop("train", {}) or {})),
            probe=ProbeConfig(**(raw.pop("probe", {}) or {})),
            **raw,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not self.stages:
            raise ValueError("a run needs at least one stage")
        if self.trace_variant not in trace.VARIANTS:
            raise ValueError(f"unknown TRACE variant {self.trace_variant!r}; "
                             f"known: {trace.VARIANTS}")
        for s in self.stages:
            if s.task not in trace.TASKS:
                raise ValueError(f"unknown task {s.task!r}; known: {trace.TASKS}")
            # Exactly one schedule. Neither is a silent zero-step stage; both is
            # ambiguous about which one actually ran.
            if (s.max_steps is None) == (s.epochs is None):
                raise ValueError(f"{s.task}: set exactly one of max_steps or epochs")
            if s.max_steps is not None and s.max_steps <= 0:
                raise ValueError(f"{s.task}: max_steps must be positive")
            if s.epochs is not None and s.epochs <= 0:
                raise ValueError(f"{s.task}: epochs must be positive")
            if s.learning_rate <= 0:
                raise ValueError(f"{s.task}: learning_rate must be positive")
        for name, val in (("probe.n_eval", self.probe.n_eval),
                          ("probe.batch_size", self.probe.batch_size),
                          ("probe.max_length", self.probe.max_length),
                          ("train.batch_size", self.train.batch_size),
                          ("train.grad_accum", self.train.grad_accum),
                          ("train.max_length", self.train.max_length),
                          ("lora.r", self.lora.r),
                          ("lora.alpha", self.lora.alpha)):
            if val <= 0:
                raise ValueError(f"{name} must be positive")
        from flab import prompts, probes as _probes
        if self.prompt_style not in prompts.STYLES:
            raise ValueError(f"prompt_style must be one of {prompts.STYLES}")
        if self.kl_scope not in _probes.KL_SCOPES:
            raise ValueError(f"kl_scope must be one of {_probes.KL_SCOPES}")
        if self.reference not in ("lima", "task_eval"):
            raise ValueError("reference must be 'lima' or 'task_eval'")
        if self.eval_split not in ("eval", "test"):
            raise ValueError("eval_split must be 'eval' or 'test'")
        if self.probe.reference_n < 0:
            raise ValueError("probe.reference_n must be >= 0 (0 disables)")
        if isinstance(self.lora.target_modules, str) and self.lora.target_modules != "all-linear":
            raise ValueError("lora.target_modules must be a list or the string 'all-linear'")
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
