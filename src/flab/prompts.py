"""Prompt rendering styles (phase-1b replication mode).

Two styles, chosen per run and recorded in the config hash because they change
every number a run produces.

**`flab`** — our own plain-text tags, shared with `data.py` so the conversion
corpus and the task corpus are formatted identically (a format difference
between them would itself be a confound in the forgetting measurement).

**`paper`** — arXiv 2606.27634's rendering, reconstructed from their published
implementation (github.com/tspthomas/slm_stability_cl, HEAD 801a3b3), not from
the paper text, which does not specify it. Their `build_messages` composes
`f"{task_prompt}\\n\\n{prompt}"` as a single user turn with **no system
message** (all their paper configs set `system_prompt: false`), rendered through
the model's own chat template with `add_generation_prompt=True`.

The task prompts are Qwen's documented defaults, which they apply to every
model, not just Qwen — so replicating means using them on Llama and Gemma too.
"""
from __future__ import annotations

from flab.data import TAGS

STYLES = ("flab", "paper")

# Verbatim from their src/constants.py.
QWEN_MATH_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
QWEN_MULTIPLE_CHOICE_PROMPT = (
    'Please show your choice in the answer field with only the choice letter, e.g., "C"'
)
TASK_PROMPT = {
    "ScienceQA": QWEN_MULTIPLE_CHOICE_PROMPT,
    "FOMC": QWEN_MULTIPLE_CHOICE_PROMPT,
    "NumGLUE-cm": QWEN_MATH_PROMPT,
}


# Llama's chat template injects "Today Date: <today>" into an automatic system
# block, so the rendered prompt — and therefore every number derived from it —
# CHANGES DAILY unless the date is pinned. That silently breaks the phase-1
# deliverable of a rig re-runnable from a commit hash: same commit, same config,
# different prompt tomorrow. Pinned to a fixed value; templates that ignore the
# kwarg are unaffected. (Found 2026-08-09 while verifying the paper-style render;
# their implementation has the same exposure.)
PINNED_DATE = "01 Jan 2026"


def _chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    """Render via the model's chat template.

    `enable_thinking` and `date_string` are passed when the template accepts
    them and dropped otherwise, since not every tokenizer's template takes them.
    """
    for extra in ({"enable_thinking": False, "date_string": PINNED_DATE},
                  {"date_string": PINNED_DATE},
                  {"enable_thinking": False},
                  {}):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt, **extra,
            )
        except TypeError:
            continue
    raise RuntimeError("apply_chat_template rejected every kwarg combination")


def render(style: str, task: str, prompt: str, answer: str, tokenizer=None) -> tuple[str, str]:
    """Return `(prefix_text, full_text)`.

    `prefix_text` must be a textual prefix of `full_text`: the answer-token
    boundary is derived as `len(tokenize(prefix))`, so anything else silently
    mislabels what is being scored. Their implementation raises if the prompt
    tokens are not a prefix of the full conversation; ours relies on the same
    property.
    """
    if style == "flab":
        prefix = f"{TAGS['user']}\n{prompt}\n{TAGS['assistant']}\n"
        return prefix, f"{prefix}{answer}\n<|end|>"

    if style != "paper":
        raise ValueError(f"unknown prompt style {style!r}; known: {STYLES}")
    if tokenizer is None:
        raise ValueError("paper style needs a tokenizer for the chat template")

    task_prompt = TASK_PROMPT.get(task)
    if task_prompt is None:
        raise ValueError(
            f"no paper task prompt for {task!r}; they define prompts only for "
            f"{sorted(TASK_PROMPT)} — using another task would silently drop the "
            "task instruction and stop being a replication")
    messages = [{"role": "user", "content": f"{task_prompt.strip()}\n\n{prompt.strip()}"}]
    prefix = _chat(tokenizer, messages, add_generation_prompt=True)
    full = _chat(tokenizer, messages + [{"role": "assistant", "content": answer.strip()}],
                 add_generation_prompt=False)
    return prefix, full
