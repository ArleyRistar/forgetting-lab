"""Load TRACE continual-learning tasks (spec §6 1a, phase-1a task 1).

TRACE has no HuggingFace mirror; `scripts/fetch_trace.sh` vendors the Drive
archive into `data/trace/`. Every task ships as three JSON files of
`{"prompt": ..., "answer": ...}`, which is what makes one loader enough.

Two things the upstream README does not say, both of which this module pins
down rather than discovers at runtime (see LAB-NOTES 2026-08-08):

  * The archive holds four variants. `_5000` is the canonical training set;
    `_500`'s 20Minuten directory is missing `eval.json` entirely, so globbing
    across variants breaks. VARIANT is fixed, not auto-detected.
  * Held-out splits range from 41 examples (NumGLUE-cm) to 2000. A requested
    `n_eval` is clamped to what exists and the real count is reported, because
    a probe silently running on 41 of a requested 200 is a number nobody can
    trust.
"""
from pathlib import Path
import hashlib
import json

from datasets import Dataset, DatasetDict

from flab import synthetic
from flab.data import TAGS

ROOT = Path(__file__).resolve().parents[2] / "data" / "trace" / "TRACE-Benchmark"
VARIANT = "LLM-CL-Benchmark_5000"

# All four ship in the archive; the README mentions none of them. `_5000` is the
# canonical training set and our default; `_500` is what arXiv 2606.27634 used,
# so phase-1b replication needs it. `_500`'s 20Minuten still lacks eval.json —
# the variant is chosen explicitly, never auto-discovered.
VARIANTS = (
    "LLM-CL-Benchmark_500",
    "LLM-CL-Benchmark_1000",
    "LLM-CL-Benchmark_5000",
    "LLM-CL-Benchmark_Reasoning",
)

# The eight training tasks. Lima is the replay set, deliberately not here.
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
    *synthetic.TASKS,
]

# Phase-1a development trio (Arley, 2026-08-08). Picked on the measured length
# profile, not the task names: classification -> code -> science QA, all
# English, all inside seq 1024. MeetingBank is excluded on purpose — 58% of its
# examples exceed the window, which makes summarization ill-posed rather than
# hard.
DEV_TASKS = ["FOMC", "Py150", "ScienceQA"]

# TRACE's replay set, used here as the stability *reference set* — disjoint from
# every task by construction, which is what 2606.27634 requires of R.
#
# Its `train` split, deliberately. **Lima/eval and Lima/test are 100% empty
# answers** — all 300 rows in each, in both variants (verified 2026-08-09) — so
# the natural reach for a held-out reference set yields zero scorable tokens.
# We never train on Lima, so its train split *is* held out for our purposes.
REFERENCE = "Lima"
REFERENCE_SPLIT = "train"

# Pre-trim ceiling. Py150 prompts reach 162k characters and tokenizing one in
# full costs far more than the ~1024 tokens that survive truncation: preparing
# Py150 took 9.1 s before this. Left-truncation keeps the END of the prompt, so
# slicing from the left first is exactly consistent with what follows. 20
# chars/token is a very generous ceiling (typical is 3-5), so the token-level
# truncation below still does the real work.
CHARS_PER_TOKEN_CEIL = 20


def pretrim(prompt: str, max_length: int) -> str:
    cap = max_length * CHARS_PER_TOKEN_CEIL
    return prompt[-cap:] if len(prompt) > cap else prompt


def available(variant: str = VARIANT) -> bool:
    """True when the vendored archive is present, so tests can skip not fail."""
    return (ROOT / variant).is_dir()


def format_example(prompt: str, answer: str) -> str:
    """Render one example in the same chat format `data.py` uses.

    Reusing `data.py`'s TAGS is deliberate: if the conversion corpus and the
    task corpus were formatted differently, the format change would itself be a
    confound in the forgetting measurement.
    """
    return f"{TAGS['user']}\n{prompt}\n{TAGS['assistant']}\n{answer}\n<|end|>"


def _truncate_prompt(prompt: str, answer: str, tokenizer, max_length: int) -> tuple[str, bool]:
    """Left-truncate the prompt so prompt+answer fits, never touching the answer.

    Answer-token NLL is the measurement, so an example whose answer was cut by
    the window is corrupted rather than noisy. Left-truncation also keeps the
    tokens immediately preceding the completion, which is the context that
    matters for Py150's long tail.
    """
    if tokenizer is None or max_length is None:
        return prompt, False

    # Budget = window minus everything that is not prompt text.
    fixed = len(tokenizer(format_example("", answer), add_special_tokens=False)["input_ids"])
    budget = max_length - fixed
    ids = tokenizer(pretrim(prompt, max_length), add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return prompt, False
    if budget <= 0:
        # The answer alone fills the window. Truncating the prompt cannot save
        # this example; surface it rather than emitting a silent empty prompt.
        return "", True
    return tokenizer.decode(ids[-budget:]), True


def prefix_of(prompt: str) -> str:
    """Everything before the answer, so a probe can find the boundary exactly.

    String-splitting a formatted example back apart would break the moment an
    answer contained the assistant tag. Building the two halves separately
    keeps the boundary exact.
    """
    return f"{TAGS['user']}\n{prompt}\n{TAGS['assistant']}\n"


def _order(n: int, seed: int) -> list[int]:
    """A deterministic permutation of range(n), stable across processes.

    Used by both the training loader and the probe. `random.shuffle` would do,
    but hashing makes the order depend only on (seed, index) — no dependence on
    interpreter version or call order, which is what "re-runnable from a commit
    hash" actually requires.
    """
    return sorted(range(n), key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def load_reference_examples(
    n: int = 200, seed: int = 0, variant: str = VARIANT, split: str | None = None
) -> tuple[list[dict], dict]:
    """Fixed reference set for stability monitoring (2606.27634's R)."""
    rows = _read(REFERENCE, split or REFERENCE_SPLIT, variant)
    take = _order(len(rows), seed)[: min(n, len(rows))]
    picked = [{"prompt": rows[i]["prompt"], "answer": rows[i]["answer"]} for i in take]
    return picked, {"requested": n, "used": len(picked), "available": len(rows)}


def load_task_eval_reference(
    tasks: list[str], fraction: float = 0.2, seed: int = 33, variant: str = VARIANT
) -> tuple[list[dict], dict]:
    """2606.27634's reference set: a slice carved out of each task's eval split.

    Their `scripts/build_reference_set.py` takes 20% of every task's
    `eval.json` (seed 33), combines and shuffles — 48 examples for the _500
    variant (20 FOMC + 20 ScienceQA + 8 NumGLUE-cm). Not a separate corpus.

    Note this makes R *task-distributed* rather than generic: it measures drift
    on the same distribution being trained on, which is a materially different
    question from drift on unrelated text (our Lima choice). Both satisfy the
    paper's stated property that R is disjoint from training data.
    """
    rows, per_task = [], {}
    for t in tasks:
        pool = _read(t, "eval", variant)
        take = _order(len(pool), seed)[: max(1, int(round(len(pool) * fraction)))]
        per_task[t] = len(take)
        rows.extend({"prompt": pool[i]["prompt"], "answer": pool[i]["answer"],
                     "task": t} for i in take)
    order = _order(len(rows), seed)
    return [rows[i] for i in order], {"used": len(rows), "per_task": per_task}


def load_probe_examples(
    name: str, n_eval: int = 200, seed: int = 0, split: str = "eval",
    variant: str = VARIANT,
) -> tuple[list[dict], dict]:
    """Held-out (prompt, answer) pairs for the NLL probe, plus honest counts.

    Kept separate from `load_task` because the probe needs the halves apart
    while training needs them joined. The selection is deterministic in `seed`,
    which is the property that matters: every boundary in a run must probe the
    *same* held-out examples, or the loss matrix is comparing different sets.
    """
    rows = _read(name, split, variant)
    take = _order(len(rows), seed)[: min(n_eval, len(rows))]
    picked = [{"prompt": rows[i]["prompt"], "answer": rows[i]["answer"]} for i in take]
    stats = {"requested": n_eval, "used": len(picked), "available": len(rows)}
    return picked, stats


def _read(task: str, split: str, variant: str = VARIANT) -> list[dict]:
    # Synthetic control tasks are generated, not vendored. They enter through
    # the same loader so the harness, probe and metrics need no special case —
    # a control that ran through a different code path would be validating a
    # different instrument than the one phase 2 uses.
    if task in synthetic.TASKS:
        return synthetic.make(task, split)
    path = ROOT / variant / task / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — run scripts/fetch_trace.sh")
    return json.loads(path.read_text())


def load_task(
    name: str,
    n_train: int | None = None,
    n_eval: int = 200,
    seed: int = 0,
    tokenizer=None,
    max_length: int | None = None,
    variant: str = VARIANT,
    prompt_style: str = "flab",
) -> DatasetDict:
    """Load one TRACE task as train/eval splits of formatted text.

    Returns a DatasetDict whose `.info` carries the counts that make the run
    auditable: how many examples were actually used, and how many prompts the
    window truncated.
    """
    if name not in TASKS:
        raise ValueError(f"unknown TRACE task {name!r}; known: {TASKS}")

    out, stats = {}, {}
    for split, want in (("train", n_train), ("eval", n_eval)):
        rows = _read(name, split, variant)
        # Select *before* formatting. Tokenizing all 5000 rows to keep a
        # handful is the dominant cost of preparing a stage, and it made the
        # CPU test suite take minutes to run two training steps. Ordering is
        # the same deterministic hash the probe uses, so `seed` still fully
        # determines which examples a stage sees.
        take = _order(len(rows), seed)
        if want is not None:
            take = take[: min(want, len(rows))]

        n_truncated, texts = 0, []
        for i in take:
            ex = rows[i]
            prompt, cut = _truncate_prompt(ex["prompt"], ex["answer"], tokenizer, max_length)
            n_truncated += cut
            if prompt_style == "flab":
                texts.append(format_example(prompt, ex["answer"]))
            else:
                from flab import prompts as _p
                texts.append(_p.render(prompt_style, name, prompt, ex["answer"], tokenizer)[1])

        out[split] = Dataset.from_dict({"text": texts})
        stats[split] = {
            "requested": want,
            "used": len(texts),
            "available": len(rows),
            # Counted over what was actually used, not over the whole file.
            "prompts_truncated": n_truncated,
        }

    dd = DatasetDict(out)
    dd.stats = stats  # type: ignore[attr-defined]
    return dd
