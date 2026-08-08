"""Held-out NLL probes — the phase-1 instrument (spec §6, phase-1a task 3).

Phase 0 settled the choice of instrument empirically: over a 400-step LoRA SFT,
every benchmark accuracy moved less than ~1.1 standard errors while held-out
loss moved clearly (1.362 -> 1.187). At this model scale accuracy is not a
usable measuring device, so forgetting is tracked with likelihood.

Three properties this module is built around, each of which would quietly
corrupt the loss matrix if dropped:

  1. **Answer tokens only.** Prompt tokens are identical at every boundary, so
     including them dilutes the signal with a large constant.
  2. **Never average NLL across tasks.** The dev trio spans 1 answer token
     (FOMC's single letter) to ~200 (ScienceQA); a cross-task mean would track
     whichever task is wordiest and would move when nothing had changed. Each
     task is only ever compared against its own baseline — which is also what
     spec §9 requires for the ternary/float comparison later.
  3. **A `warning` field that can say "I did not measure this."** The first
     memory probe returned a plausible `0.0` because its callback was never
     registered; an explicit warning is what caught it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time

import torch
import torch.nn.functional as F

from flab import trace

IGNORE = -100
BUCKET = 128  # pad batch widths to a multiple of this; see probe_task


def _bucket(n: int) -> int:
    return ((n + BUCKET - 1) // BUCKET) * BUCKET


@dataclass
class TaskProbe:
    task: str
    nll: float | None          # mean per-token NLL over answer tokens
    token_acc: float | None
    n_tokens: int
    n_examples: int
    n_prompt_truncated: int
    n_dropped: int
    seconds: float
    warning: str | None


def ensure_no_live_adapter(model) -> None:
    """Refuse generative evaluation on an unmerged PEFT model.

    Measured 2026-08-07: an unmerged adapter costs **1.89x** on generative
    decode, because the per-layer LoRA matmuls land on the critical path of
    every token instead of being absorbed into a large training matmul. The
    harness evaluates at every stage boundary, so paying that repeatedly would
    be a large fraction of the phase's budget. Merge into a throwaway copy
    (`merge_and_unload`) first.

    This is a guard, not a suggestion — hence a raise rather than a comment.
    Likelihood probes are unaffected and do not call this.
    """
    if hasattr(model, "peft_config") or type(model).__name__.startswith("Peft"):
        raise RuntimeError(
            "generative eval on a live PEFT model costs 1.89x (measured "
            "2026-08-07); call merge_and_unload() on a copy first"
        )


def _encode(tokenizer, prompt: str, answer: str, max_length: int):
    """Build (input_ids, labels) with an exact prompt/answer boundary.

    The two halves are tokenized separately and concatenated rather than
    tokenizing the joined string and guessing where the answer starts: a
    tokenizer may merge across the join, which would silently shift the
    boundary by a token and mislabel what is being scored. A probe only has to
    be *consistent across stages*, and building from parts guarantees that.

    Prompts are left-truncated; the answer is never touched, because
    answer-token NLL is the measurement and a cut answer is a corrupted data
    point rather than a noisy one.
    """
    prompt = trace.pretrim(prompt, max_length)
    prefix_ids = tokenizer(trace.prefix_of(prompt), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    # Need at least one prompt token and one answer token to score anything.
    if len(answer_ids) + 1 >= max_length:
        return None, None, False
    budget = max_length - len(answer_ids)
    truncated = len(prefix_ids) > budget
    if truncated:
        prefix_ids = prefix_ids[-budget:]

    return prefix_ids + answer_ids, [IGNORE] * len(prefix_ids) + answer_ids, truncated


@torch.no_grad()
def probe_task(
    model,
    tokenizer,
    task: str,
    n_eval: int = 200,
    max_length: int = 1024,
    batch_size: int = 4,
    seed: int = 0,
) -> TaskProbe:
    """Mean per-token NLL and token accuracy over one task's held-out answers."""
    t0 = time.perf_counter()
    examples, stats = trace.load_probe_examples(task, n_eval=n_eval, seed=seed)

    encoded, n_trunc, n_dropped = [], 0, 0
    for ex in examples:
        ids, labels, truncated = _encode(tokenizer, ex["prompt"], ex["answer"], max_length)
        if ids is None:
            n_dropped += 1
            continue
        n_trunc += truncated
        encoded.append((ids, labels))

    if not encoded:
        return TaskProbe(task, None, None, 0, 0, n_trunc, n_dropped, time.perf_counter() - t0,
                         "no example survived encoding; nothing was measured")

    # Length-sorted batching: pure padding saving, and order-independent since
    # the aggregate is a sum over tokens.
    encoded.sort(key=lambda p: len(p[0]))
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id or 0
    was_training = model.training
    model.eval()

    nll_sum, correct, n_tokens = 0.0, 0, 0
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        # Round the width up to a bucket. Length-sorted batching otherwise
        # produces a near-unique width per batch, and the caching allocator
        # hoards a block per distinct shape: measured 7424 MiB *reserved*
        # against 2321 MiB allocated (3.2x) before bucketing, which is 97% of
        # the card for a forward-only pass. Budget is against reserved.
        width = min(max_length, _bucket(max(len(ids) for ids, _ in chunk)))
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        labels = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, (ids, lab) in enumerate(chunk):
            input_ids[r, : len(ids)] = torch.tensor(ids)
            labels[r, : len(lab)] = torch.tensor(lab)
            attn[r, : len(ids)] = 1

        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=input_ids, attention_mask=attn).logits

        # Standard causal shift: position t predicts token t+1.
        shift_labels = labels[:, 1:]
        mask = shift_labels != IGNORE
        if not mask.any():
            continue
        # Gather the scored positions *before* upcasting. Upcasting first would
        # materialise batch x seq x 49k in fp32 (~800 MiB at batch 4) to then
        # use a handful of rows — and on FOMC that is one row per example.
        # Activation memory here is logit-dominated (LAB-NOTES 2026-08-07).
        flat_logits = logits[:, :-1, :][mask].float()
        flat_labels = shift_labels[mask]
        nll_sum += F.cross_entropy(flat_logits, flat_labels, reduction="sum").item()
        correct += (flat_logits.argmax(-1) == flat_labels).sum().item()
        n_tokens += int(mask.sum())

    if was_training:
        model.train()

    warning = None
    if n_tokens == 0:
        warning = "zero answer tokens scored; the NLL below is not a measurement"
    elif n_dropped:
        warning = f"{n_dropped} example(s) dropped: answer alone exceeded max_length"
    elif stats["used"] < stats["requested"]:
        warning = (f"only {stats['used']} of {stats['requested']} requested examples exist "
                   f"for {task}")

    return TaskProbe(
        task=task,
        nll=(nll_sum / n_tokens) if n_tokens else None,
        token_acc=(correct / n_tokens) if n_tokens else None,
        n_tokens=n_tokens,
        n_examples=len(encoded),
        n_prompt_truncated=n_trunc,
        n_dropped=n_dropped,
        seconds=time.perf_counter() - t0,
        warning=warning,
    )


def probe_all(model, tokenizer, tasks: list[str], *, n_eval: int = 200,
              max_length: int = 1024, batch_size: int = 4, seed: int = 0) -> dict:
    """Probe every task, returning a dict ready to serialise to a boundary file.

    Every task is probed at every boundary, including tasks not yet trained on:
    the resulting N x N matrix yields forgetting, backward transfer *and*
    forward transfer with no further runs. Probing only seen tasks would throw
    the forward-transfer half away for a saving not worth having.

    There is deliberately no aggregate NLL across tasks — see the module
    docstring. `tasks` is a dict keyed by task name so no ordering is implied.
    """
    # The probe runs immediately after a training stage, whose allocation is
    # still cached. Without this the probe inherits that reservation on top of
    # its own and can OOM on a forward-only pass.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    results = {t: probe_task(model, tokenizer, t, n_eval=n_eval, max_length=max_length,
                             batch_size=batch_size, seed=seed) for t in tasks}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    warnings = {t: r.warning for t, r in results.items() if r.warning}
    return {
        "tasks": {t: asdict(r) for t, r in results.items()},
        "seconds_total": sum(r.seconds for r in results.values()),
        "warnings": warnings or None,
    }
