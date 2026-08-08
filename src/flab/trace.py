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
import json

from datasets import Dataset, DatasetDict

from flab.data import TAGS

ROOT = Path(__file__).resolve().parents[2] / "data" / "trace" / "TRACE-Benchmark"
VARIANT = "LLM-CL-Benchmark_5000"

# The eight training tasks. Lima is the replay set, deliberately not here.
TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]

# Phase-1a development trio (Arley, 2026-08-08). Picked on the measured length
# profile, not the task names: classification -> code -> science QA, all
# English, all inside seq 1024. MeetingBank is excluded on purpose — 58% of its
# examples exceed the window, which makes summarization ill-posed rather than
# hard.
DEV_TASKS = ["FOMC", "Py150", "ScienceQA"]


def available() -> bool:
    """True when the vendored archive is present, so tests can skip not fail."""
    return (ROOT / VARIANT).is_dir()


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
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return prompt, False
    if budget <= 0:
        # The answer alone fills the window. Truncating the prompt cannot save
        # this example; surface it rather than emitting a silent empty prompt.
        return "", True
    return tokenizer.decode(ids[-budget:]), True


def _read(task: str, split: str) -> list[dict]:
    path = ROOT / VARIANT / task / f"{split}.json"
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
        rows = _read(name, split)
        n_truncated = 0
        texts = []
        for ex in rows:
            prompt, cut = _truncate_prompt(ex["prompt"], ex["answer"], tokenizer, max_length)
            n_truncated += cut
            texts.append(format_example(prompt, ex["answer"]))
        ds = Dataset.from_dict({"text": texts}).shuffle(seed=seed)
        # Clamp rather than fail: NumGLUE-cm only has 41 eval examples.
        if want is not None:
            ds = ds.select(range(min(want, len(ds))))
        out[split] = ds
        stats[split] = {
            "requested": want,
            "used": len(ds),
            "available": len(rows),
            "prompts_truncated": n_truncated,
        }

    dd = DatasetDict(out)
    dd.stats = stats  # type: ignore[attr-defined]
    return dd
