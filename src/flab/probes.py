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


def _encode(tokenizer, prompt: str, answer: str, max_length: int,
            allow_answer_truncation: bool = False):
    """Build (input_ids, labels) with an exact prompt/answer boundary.

    The two halves are tokenized separately and concatenated rather than
    tokenizing the joined string and guessing where the answer starts: a
    tokenizer may merge across the join, which would silently shift the
    boundary by a token and mislabel what is being scored. A probe only has to
    be *consistent across stages*, and building from parts guarantees that.

    Prompts are left-truncated; the answer is never touched, because
    answer-token NLL is the measurement and a cut answer is a corrupted data
    point rather than a noisy one.

    `allow_answer_truncation` exists for the **stability probe only**, and the
    asymmetry is deliberate rather than a convenience. KL compares two
    distributions over *identical* inputs, so both the base and the current
    model see exactly the same (possibly truncated) tokens and the comparison
    stays valid — truncation changes which tokens are sampled, not what the
    number means. For NLL it would corrupt the measurement outright, scoring
    the likelihood of a fragment as though it were a whole answer.

    Without it the reference set is unusable: Lima answers run to a median of
    **375 tokens and a p90 of 1602**, so at the paper's 512-token window only
    66% fit at all — and dropping the rest would silently bias the reference
    set toward short-answer examples (measured 2026-08-09).
    """
    prompt = trace.pretrim(prompt, max_length)
    prefix_ids = tokenizer(trace.prefix_of(prompt), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if allow_answer_truncation:
        # Cap the answer at 3/4 of the window so real prompt context survives.
        answer_ids = answer_ids[: max(1, (max_length * 3) // 4)]
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
    variant: str = trace.VARIANT,
) -> TaskProbe:
    """Mean per-token NLL and token accuracy over one task's held-out answers."""
    t0 = time.perf_counter()
    examples, stats = trace.load_probe_examples(task, n_eval=n_eval, seed=seed, variant=variant)

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
              max_length: int = 1024, batch_size: int = 4, seed: int = 0,
              variant: str = trace.VARIANT) -> dict:
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
                             batch_size=batch_size, seed=seed, variant=variant) for t in tasks}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    warnings = {t: r.warning for t, r in results.items() if r.warning}
    return {
        "tasks": {t: asdict(r) for t, r in results.items()},
        "seconds_total": sum(r.seconds for r in results.values()),
        "warnings": warnings or None,
    }


# -- stability / drift (phase-1b task 3) ---------------------------------


@dataclass
class StabilityProbe:
    """Distributional drift from the base model on a fixed reference set.

    This is 2606.27634's stability half, and it matters beyond the replication:
    **KL from base on a held-out set is the float-side analogue of phase 1d's
    ternary flip-fraction** — both ask "how far has this model moved", one in
    output space and one in weight space. Building it here means phase 2 gets to
    compare like with like rather than inventing a bridge later.
    """
    n_tokens: int
    n_examples: int
    kl_from_base: float | None      # KL(p_base || p_current), nats/token
    delta_entropy: float | None     # H(p_current) - H(p_base)
    margin: float | None            # top-1 minus top-2 log-prob, current
    base_margin: float | None
    seconds: float
    warning: str | None


@torch.no_grad()
def probe_stability(
    model,
    tokenizer,
    n_ref: int = 200,
    max_length: int = 512,
    batch_size: int = 2,
    seed: int = 0,
    variant: str = trace.VARIANT,
    base_model=None,
    split: str | None = None,
) -> StabilityProbe:
    """Compare the current model against its own base on the reference set.

    Under LoRA the base distribution is obtained by *disabling the adapter*
    rather than holding a second copy of the weights: it is the same tensors by
    construction, and costs no extra VRAM on a 7.5 GiB card. A full fine-tune
    has no such trick, so `base_model` may be passed explicitly — phase 1c will
    need that.
    """
    t0 = time.perf_counter()
    examples, stats = trace.load_reference_examples(
        n=n_ref, seed=seed, variant=variant, split=split)

    encoded = []
    for ex in examples:
        ids, labels, _ = _encode(tokenizer, ex["prompt"], ex["answer"], max_length,
                                 allow_answer_truncation=True)
        if ids is not None:
            encoded.append((ids, labels))

    can_disable = hasattr(model, "disable_adapter")
    if not encoded or not (can_disable or base_model is not None):
        why = ("no example survived encoding" if not encoded else
               "model is not a PEFT model and no base_model was given")
        return StabilityProbe(0, 0, None, None, None, None, time.perf_counter() - t0,
                              f"{why}; nothing was measured")

    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id or 0
    was_training = model.training
    model.eval()
    encoded.sort(key=lambda p: len(p[0]))

    kl_sum = dh_sum = m_sum = bm_sum = 0.0
    n_tokens = 0
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        width = min(max_length, _bucket(max(len(ids) for ids, _ in chunk)))
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        labels = torch.full((len(chunk), width), IGNORE, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, (ids, lab) in enumerate(chunk):
            input_ids[r, : len(ids)] = torch.tensor(ids)
            labels[r, : len(lab)] = torch.tensor(lab)
            attn[r, : len(ids)] = 1
        input_ids, labels, attn = input_ids.to(device), labels.to(device), attn.to(device)

        mask = labels[:, 1:] != IGNORE
        if not mask.any():
            continue

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            cur = model(input_ids=input_ids, attention_mask=attn).logits
            if base_model is not None:
                base = base_model(input_ids=input_ids, attention_mask=attn).logits
            else:
                with model.disable_adapter():
                    base = model(input_ids=input_ids, attention_mask=attn).logits

        # Gather scored positions before upcasting — the same reason as in
        # probe_task: full batch x seq x vocab in fp32 is ~800 MiB at 49k vocab
        # and 2.6x that on Llama's 128k.
        lp_cur = F.log_softmax(cur[:, :-1, :][mask].float(), dim=-1)
        lp_base = F.log_softmax(base[:, :-1, :][mask].float(), dim=-1)
        p_base = lp_base.exp()

        kl_sum += (p_base * (lp_base - lp_cur)).sum(-1).sum().item()
        dh_sum += (-(lp_cur.exp() * lp_cur).sum(-1) + (p_base * lp_base).sum(-1)).sum().item()
        m_sum += (lambda t: (t[:, 0] - t[:, 1]).sum().item())(lp_cur.topk(2, -1).values)
        bm_sum += (lambda t: (t[:, 0] - t[:, 1]).sum().item())(lp_base.topk(2, -1).values)
        n_tokens += int(mask.sum())

    if was_training:
        model.train()

    if n_tokens == 0:
        return StabilityProbe(0, len(encoded), None, None, None, None,
                              time.perf_counter() - t0,
                              "zero reference tokens scored; nothing was measured")

    return StabilityProbe(
        n_tokens=n_tokens,
        n_examples=len(encoded),
        kl_from_base=kl_sum / n_tokens,
        delta_entropy=dh_sum / n_tokens,
        margin=m_sum / n_tokens,
        base_margin=bm_sum / n_tokens,
        seconds=time.perf_counter() - t0,
        warning=(None if stats["used"] == stats["requested"]
                 else f"only {stats['used']} of {stats['requested']} reference examples exist"),
    )
